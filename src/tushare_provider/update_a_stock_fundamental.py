import os
import time
import datetime
from typing import Optional, List

import fire
import pandas
import tushare as ts
from sqlalchemy import create_engine
from sqlalchemy.types import Float, DECIMAL, String
import pymysql  # noqa: F401 - required by SQLAlchemy URL


ts.set_token(os.environ["TUSHARE"])
pro = ts.pro_api()


def get_trade_cal(start_date: str, end_date: str) -> pandas.DataFrame:
    df = pro.trade_cal(
        exchange="SSE",
        is_open="1",
        start_date=start_date,
        end_date=end_date,
        fields="cal_date",
    )
    return df


def get_daily_basic(trade_date: str) -> Optional[pandas.DataFrame]:
    fields = (
        "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,"
        "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,"
        "total_share,float_share,free_share,total_mv,circ_mv"
    )
    for _ in range(3):
        try:
            df = pro.daily_basic(trade_date=trade_date, fields=fields)
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
        # "trade_date": handled as date object separately
        "turnover_rate": Float(),
        "turnover_rate_f": Float(),
        "volume_ratio": Float(),
        "pe": Float(),
        "pe_ttm": Float(),
        "pb": Float(),
        "ps": Float(),
        "ps_ttm": Float(),
        "dv_ratio": Float(),
        "dv_ttm": Float(),
        "total_share": DECIMAL(16, 4),
        "float_share": DECIMAL(16, 4),
        "free_share": DECIMAL(16, 4),
        "total_mv": DECIMAL(16, 4),
        "circ_mv": DECIMAL(16, 4),
    }

    written = batch_df.to_sql(
        "ts_a_stock_fundamental",
        sql_engine,
        if_exists="append",
        index=False,
        chunksize=chunksize,
        method="multi",
        dtype=dtype,
    )
    pending_frames.clear()
    return int(written or 0)


def update_astock_fundamental_to_latest(
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

    sql_engine = create_engine(
        "mysql+pymysql://root:@127.0.0.1:3307/investment_data_new", pool_recycle=3600
    )
    db_conn = sql_engine.raw_connection()

    # Determine latest fully populated trade_date present
    latest_sql = (
        """
        select max(trade_date) as trade_date
        from (
            select trade_date, count(1) as symbol_count
            from ts_a_stock_fundamental
            where trade_date > "20250501" 
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
        latest_trade_date = "20080101"

    latest_trade_date = latest_trade_date.replace("-", "")
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
        print("Downloading fundamentals", trade_date)
        data = get_daily_basic(trade_date)
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
    fire.Fire(update_astock_fundamental_to_latest)

