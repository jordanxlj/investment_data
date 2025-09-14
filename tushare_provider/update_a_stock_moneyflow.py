import os
import time
import datetime
from typing import Optional, List

import fire
import pandas
import tushare as ts
from sqlalchemy import create_engine, text
from sqlalchemy.types import Float, DECIMAL, String
import pymysql  # noqa: F401 - required by SQLAlchemy URL


ts.set_token(os.environ["TUSHARE"])
pro = ts.pro_api()

TABLE_NAME = "ts_a_stock_moneyflow"

CREATE_TABLE_DDL = f"""
CREATE TABLE  IF NOT EXISTS {TABLE_NAME}  (
   ts_code  varchar(16) NOT NULL,
   trade_date  varchar(8) NOT NULL,
   buy_sm_amount  FLOAT NULL,
   buy_md_amount  FLOAT NULL,
   buy_lg_amount  FLOAT NULL,
   buy_elg_amount  FLOAT NULL,
   sell_sm_amount  FLOAT NULL,
   sell_md_amount  FLOAT NULL,
   sell_lg_amount  FLOAT NULL,
   sell_elg_amount  FLOAT NULL,
   net_mf_amount  FLOAT NULL,
  PRIMARY KEY ( ts_code , trade_date ),
  INDEX idx_ts_code (ts_code)
  INDEX idx_trade_date (trade_date),
  INDEX idx_ts_code_trade_date (ts_code, trade_date),
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;
"""

def get_trade_cal(start_date: str, end_date: str) -> pandas.DataFrame:
    df = pro.trade_cal(
        exchange="SSE",
        is_open="1",
        start_date=start_date,
        end_date=end_date,
        fields="cal_date",
    )
    return df


def get_moneyflow(trade_date: str) -> Optional[pandas.DataFrame]:
    fields = (
        "ts_code,trade_date,buy_sm_amount,buy_md_amount,buy_lg_amount,"
        "buy_elg_amount,sell_sm_amount,sell_md_amount,sell_lg_amount,"
        "sell_elg_amount,net_mf_amount"
    )
    for _ in range(3):
        try:
            df = pro.moneyflow(trade_date=trade_date, fields=fields)
            return df
        except Exception as e:
            print(e)
            time.sleep(1)
    return None


def _flush_batch(
    sql_engine, pending_frames: List[pandas.DataFrame], chunksize: int = 2000
) -> int:
    if not pending_frames:
        return 0
    batch_df = pandas.concat(pending_frames, ignore_index=True)
    if batch_df.empty:
        pending_frames.clear()
        return 0
    # Persist with multi-row insert for performance
    # enforce dtypes for performance/stability
    dtype = {
        "ts_code": String(16),
        "trade_date": String(8),
        "buy_sm_amount": Float(),
        "buy_md_amount": Float(),
        "buy_lg_amount": Float(),
        "buy_elg_amount": Float(),
        "sell_sm_amount": Float(),
        "sell_md_amount": Float(),
        "sell_lg_amount": Float(),
        "sell_elg_amount": Float(),
        "net_mf_amount": Float(),
    }

    written = batch_df.to_sql(
        "ts_a_stock_moneyflow",
        sql_engine,
        if_exists="append",
        index=False,
        chunksize=chunksize,
        method="multi",
        dtype=dtype,
    )
    pending_frames.clear()
    return int(written or 0)


def update_astock_moneyflow_to_latest(
    mysql_url: str = "mysql+pymysql://root:@127.0.0.1:3306/investment_data",
    min_symbols_per_day: int = 1000,
    max_days_per_batch: int = 30,
    max_rows_per_batch: int = 50000,
    chunksize: int = 2000,
) -> None:
    """Append missing ts_a_stock_fundamental rows to MySQL from Tushare daily_basic.

    - Finds latest trade_date present in table
    - Iterates subsequent open days up to today
    - Fetches daily_basic and appends to table
    """

    sql_engine = create_engine(mysql_url, pool_recycle=3600)
    with sql_engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_DDL))

    db_conn = sql_engine.raw_connection()

    # Determine latest fully populated trade_date present
    latest_sql = (
        """
        select max(trade_date) as trade_date
        from (
            select trade_date, count(1) as symbol_count
            from ts_a_stock_moneyflow
            group by trade_date
        ) t
        where symbol_count > {min_symbols}
        """
    ).format(min_symbols=min_symbols_per_day)

    latest_trade_date: Optional[str] = None
    try:
        latest_df = pandas.read_sql(latest_sql, db_conn)
        val = latest_df["trade_date"][0]
        if pandas.notna(val):
            latest_trade_date = str(val)
    except Exception as e:
        print("Failed to read latest trade_date:", e)

    if not latest_trade_date:
        latest_trade_date = "20100101"

    end_date = datetime.datetime.now().strftime("%Y%m%d")

    trade_date_df = get_trade_cal(latest_trade_date, end_date)
    trade_date_df = trade_date_df.sort_values("cal_date").reset_index(drop=True)

    pending_frames: List[pandas.DataFrame] = []
    pending_rows = 0
    pending_days = 0

    for row in trade_date_df.values.tolist():
        trade_date = row[0]
        if trade_date == latest_trade_date:
            continue
        print("Downloading moneyflow", trade_date)
        data = get_moneyflow(trade_date)
        if data is None or data.empty:
            continue

        pending_frames.append(data)
        pending_rows += len(data)
        pending_days += 1

        if pending_days >= max_days_per_batch or pending_rows >= max_rows_per_batch:
            written = _flush_batch(sql_engine, pending_frames, chunksize=chunksize)
            print(
                f"Flushed batch: days={pending_days}, rows~={pending_rows}, written={written}"
            )
            pending_rows = 0
            pending_days = 0

    # Final flush
    if pending_frames:
        written = _flush_batch(sql_engine, pending_frames, chunksize=chunksize)
        print(f"Final flush written={written}")


if __name__ == "__main__":
    fire.Fire(update_astock_moneyflow_to_latest)

