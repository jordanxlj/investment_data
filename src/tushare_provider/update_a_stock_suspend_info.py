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

TABLE_NAME = "ts_a_stock_suspend_info"

CREATE_TABLE_DDL = f"""
CREATE TABLE  IF NOT EXISTS {TABLE_NAME}  (
   ts_code  varchar(16) NOT NULL,
   trade_date  DATE NOT NULL,
   suspend_timing  varchar(8) NOT NULL,
   suspend_type  varchar(1) NOT NULL,
  PRIMARY KEY ( ts_code , trade_date ),
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


def get_suspend(trade_date: str) -> Optional[pandas.DataFrame]:
    fields = (
        "ts_code,trade_date,suspend_timing,suspend_type"
    )
    for _ in range(3):
        try:
            df = pro.suspend_d(trade_date=trade_date, fields=fields)
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
    # Note: trade_date is converted to date object in data processing
    # and stored as DATE type in MySQL, so we don't specify dtype for it
    dtype = {
        "ts_code": String(16),
        # "trade_date": not specified - handled as date object separately
        "suspend_timing": String(8),
        "suspend_type": String(1),
    }

    # Use INSERT IGNORE to skip duplicates without updating
    def insert_ignore_method(table, conn, keys, data_iter):
        data = [dict(zip(keys, row)) for row in data_iter]
        if not data:
            return
        # Build INSERT IGNORE statement - trade_date is already a proper date object
        columns = ', '.join(f'`{k}`' for k in keys)
        values_placeholders = ', '.join(f':{k}' for k in keys)
        insert_stmt = f"INSERT IGNORE INTO `{table.table.name}` ({columns}) VALUES ({values_placeholders})"

        # Execute in batches
        from sqlalchemy.sql import text
        for row in data:
            try:
                conn.execute(text(insert_stmt), row)
            except Exception as e:
                print(f"Skip conflict row: {row.get('ts_code', 'unknown')}-{row.get('trade_date', 'unknown')}, error: {e}")

    written = batch_df.to_sql(
        "ts_a_stock_suspend_info",
        sql_engine,
        if_exists="append",
        index=False,
        chunksize=chunksize,
        method=insert_ignore_method,
        dtype=dtype,
    )
    pending_frames.clear()
    return int(written or 0)


def update_astock_suspend_to_latest(
    mysql_url: str = "mysql+pymysql://root:@127.0.0.1:3306/investment_data",
    min_symbols_per_day: int = 1000,
    max_days_per_batch: int = 30,
    max_rows_per_batch: int = 50000,
    chunksize: int = 2000,
) -> None:
    """Append missing ts_a_stock_suspend_info rows to MySQL from Tushare suspend data.
    Process:
    - Finds latest trade_date present in table
    - Iterates subsequent open days up to today
    - Fetches suspend data and converts trade_date to date objects
    - Inserts data with optimized batch processing
    """

    sql_engine = create_engine(mysql_url, pool_recycle=3600)
    with sql_engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_DDL))

    db_conn = sql_engine.raw_connection()

    # Determine latest trade_date present (no symbol threshold; suspend rows per day may be sparse)
    latest_trade_date: Optional[str] = None
    try:
        with sql_engine.begin() as econn:
            latest_df = pandas.read_sql_query(
                "SELECT DATE_FORMAT(MAX(trade_date), '%Y%m%d') AS trade_date FROM ts_a_stock_suspend_info",
                econn,
            )
            val = latest_df["trade_date"].iloc[0] if not latest_df.empty else None
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
        print("Downloading suspend", trade_date)
        data = get_suspend(trade_date)
        if data is None or data.empty:
            continue

        # Convert trade_date from string to date object for better performance
        if 'trade_date' in data.columns:
            data['trade_date'] = pandas.to_datetime(data['trade_date'], format='%Y%m%d').dt.date

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
    fire.Fire(update_astock_suspend_to_latest)

