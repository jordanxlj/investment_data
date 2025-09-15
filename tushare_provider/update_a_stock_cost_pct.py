import os
import time
import datetime
from typing import Optional, List, Dict, Any

import fire
import pandas as pd
import tushare as ts
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
import pymysql  # noqa: F401 - required by SQLAlchemy URL


# Initialize Tushare
ts.set_token(os.environ["TUSHARE"])  # expects env var set
pro = ts.pro_api()


TABLE_NAME = "ts_a_stock_cost_pct"


CREATE_TABLE_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
  ts_code     VARCHAR(16) NOT NULL,
  trade_date  DATE        NOT NULL,
  cost_5pct   FLOAT NULL,
  cost_15pct  FLOAT NULL,
  cost_50pct  FLOAT NULL,
  cost_85pct  FLOAT NULL,
  cost_95pct  FLOAT NULL,
  weight_avg  FLOAT NULL,
  winner_rate FLOAT NULL,
  PRIMARY KEY (ts_code, trade_date),
  KEY idx_trade_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;
"""


ALL_COLUMNS: List[str] = [
    "ts_code",
    "trade_date",
    "cost_5pct",
    "cost_15pct",
    "cost_50pct",
    "cost_85pct",
    "cost_95pct",
    "weight_avg",
    "winner_rate",
]

def get_trade_cal(start_date: str, end_date: str) -> pd.DataFrame:
    df = pro.trade_cal(
        exchange="SSE",
        is_open="1",
        start_date=start_date,
        end_date=end_date,
        fields="cal_date",
    )
    return df

def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure all expected columns exist
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    out = df[ALL_COLUMNS].copy()
    if not out.empty:
        # Convert trade_date from string to DATE object for efficient storage and queries
        # This avoids SQL-level STR_TO_DATE() conversion and improves insertion performance
        out["trade_date"] = pd.to_datetime(out["trade_date"], format='%Y%m%d', errors='coerce').dt.date
        # Numeric coercion
        for c in [
            "cost_5pct",
            "cost_15pct",
            "cost_50pct",
            "cost_85pct",
            "cost_95pct",
            "weight_avg",
            "winner_rate",
        ]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _fetch_day_with_retry(trade_date: str, retries: int = 3, sleep_secs: float = 0.3) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for _ in range(max(1, retries)):
        try:
            df = pro.cyq_perf(trade_date=trade_date, limit=5000)
            return df if df is not None else pd.DataFrame(columns=ALL_COLUMNS)
        except Exception as e:
            last_err = e
            print("tushare error, retrying:", e)
            time.sleep(sleep_secs)
    print("tushare failed after retries:", last_err)
    return pd.DataFrame(columns=ALL_COLUMNS)


def _upsert_batch(engine, df: pd.DataFrame, chunksize: int = 2000) -> int:
    if df is None or df.empty:
        return 0
    total = 0
    from sqlalchemy import Table, MetaData
    meta = MetaData()
    table = Table(TABLE_NAME, meta, autoload_with=engine)

    rows = df.to_dict(orient="records")
    with engine.begin() as conn:
        for i in range(0, len(rows), chunksize):
            batch = rows[i:i+chunksize]
            stmt = mysql_insert(table).values(batch)
            update_map: Dict[str, Any] = {
                c: getattr(stmt.inserted, c)
                for c in ALL_COLUMNS
                if c not in ("ts_code", "trade_date")
            }
            ondup = stmt.on_duplicate_key_update(**update_map)
            result = conn.execute(ondup)
            total += result.rowcount or 0
    return total


def update_a_stock_cost_pct(
    mysql_url: str = "mysql+pymysql://root:@127.0.0.1:3306/investment_data",
    start_date_override: Optional[str] = None,
    end_date: Optional[str] = None,
    page_limit: int = 5000,
    sleep_secs: float = 0.2,
    chunksize: int = 2000,
) -> None:
    """
    Ingest Tushare cyq_perf (daily chip cost percentiles & win rate) into MySQL.

    Process:
    - Auto-creates table with DATE type trade_date field
    - Incremental by max(trade_date) unless start_date_override provided
    - Uses paginated range fetch for efficiency and upsert for idempotency
    """

    engine = create_engine(mysql_url, pool_recycle=3600)
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_DDL))

    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y%m%d")

    # Determine start
    start_date: Optional[str] = None
    if start_date_override and len(start_date_override) == 8:
        start_date = start_date_override
    else:
        with engine.begin() as conn:
            res = conn.execute(text(f"SELECT DATE_FORMAT(MAX(trade_date), '%Y%m%d') FROM {TABLE_NAME}"))
            row = res.fetchone()
            if row and row[0]:
                # next day after max
                cur = datetime.datetime.strptime(str(row[0]), "%Y%m%d")
                start_date = (cur + datetime.timedelta(days=1)).strftime("%Y%m%d")
    if not start_date:
        start_date = "20180101"  # per Tushare, data from 2018

    # Iterate trading calendar and fetch per day with retries
    cal = get_trade_cal(start_date, end_date)
    cal = cal.sort_values("cal_date").reset_index(drop=True)
    total_written = 0
    for row in cal.values.tolist():
        trade_date = row[0]
        print(f"Downloading cyq_perf {trade_date}")
        raw = _fetch_day_with_retry(trade_date, retries=3, sleep_secs=sleep_secs)
        if raw is None or raw.empty:
            continue
        df = _coerce_schema(raw)
        df = df.drop_duplicates(subset=["ts_code", "trade_date"])  # safety
        written = _upsert_batch(engine, df, chunksize=chunksize)
        total_written += written
        print(f"{trade_date} upserted rows ~= {written}")
    print(f"Completed. Total rows upserted ~= {total_written}")


if __name__ == "__main__":
    fire.Fire(update_a_stock_cost_pct)
