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


# Tushare init
ts.set_token(os.environ["TUSHARE"])  # expects env var set
pro = ts.pro_api()


TABLE_NAME = "ts_a_stock_brokerage_report"


CREATE_TABLE_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
  ts_code      VARCHAR(16)  NOT NULL,
  report_date  VARCHAR(8)   NOT NULL,
  report_title VARCHAR(128) NULL,
  report_type  VARCHAR(16)  NULL,
  classify     VARCHAR(16)  NULL,
  org_name     VARCHAR(32)  NOT NULL,
  quarter      VARCHAR(16)  NULL,
  rating       VARCHAR(16)  NULL,
  op_rt        FLOAT NULL,
  op_pr        FLOAT NULL,
  tp           FLOAT NULL,
  np           FLOAT NULL,
  eps          FLOAT NULL,
  pe           FLOAT NULL,
  rd           FLOAT NULL,
  roe          FLOAT NULL,
  ev_ebitda    FLOAT NULL,
  max_price    FLOAT NULL,
  min_price    FLOAT NULL,
  PRIMARY KEY (ts_code, report_date, org_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;
"""


ALL_COLUMNS: List[str] = [
    "ts_code", "report_date", "report_title", "report_type", "classify",
    "org_name", "quarter", "op_rt", "op_pr", "tp", "np",
    "eps", "pe", "rd", "roe", "ev_ebitda", "rating", "max_price", "min_price", 
]


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure all expected columns exist
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    # Keep only known columns, in order
    out = df[ALL_COLUMNS].copy()
    # Normalize types
    if not out.empty:
        # Datetime-like string to YYYY-MM-DD HH:MM:SS
        if "create_time" in out.columns:
            out["create_time"] = pd.to_datetime(out["create_time"], errors="coerce")
        # Ensure numeric columns coerced
        for c in ["op_rt","op_pr","tp","np","eps","pe","rd","roe","ev_ebitda","max_price","min_price"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        # report_date keep as YYYYMMDD string
        out["report_date"] = out["report_date"].astype(str).str.replace("-", "").str.slice(0, 8)
        # Trim long strings
        out["report_title"] = out["report_title"].astype(str).str.slice(0, 512)
        out["org_name"] = out["org_name"].astype(str).str.slice(0, 128)
        out["rating"] = out["rating"].astype(str).str.slice(0, 64)
        out["report_type"] = out["report_type"].astype(str).str.slice(0, 64)
        out["classify"] = out["classify"].astype(str).str.slice(0, 64)
        out["quarter"] = out["quarter"].astype(str).str.slice(0, 16)
    return out


def _fetch_day(report_date: str, page_limit: int = 3000, sleep_secs: float = 0.3) -> pd.DataFrame:
    all_parts: List[pd.DataFrame] = []
    offset = 0
    while True:
        try:
            df = pro.report_rc(report_date=report_date, limit=page_limit, offset=offset)
        except Exception as e:
            print("tushare error:", e)
            time.sleep(1)
            continue
        if df is None or df.empty:
            break
        all_parts.append(df)
        if len(df) < page_limit:
            break
        offset += page_limit
        time.sleep(sleep_secs)
    if not all_parts:
        return pd.DataFrame(columns=ALL_COLUMNS)
    return pd.concat(all_parts, ignore_index=True)


def _upsert_batch(engine, df: pd.DataFrame, chunksize: int = 1000) -> int:
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
                if c not in ("ts_code", "report_date", "org_name")
            }
            ondup = stmt.on_duplicate_key_update(**update_map)
            result = conn.execute(ondup)
            total += result.rowcount or 0
    return total


def update_a_stock_brokerage_report(
    mysql_url: str = "mysql+pymysql://root:@127.0.0.1:3306/investment_data",
    start_date_override: Optional[str] = None,
    end_date: Optional[str] = None,
    page_limit: int = 3000,
    sleep_secs: float = 0.3,
) -> None:
    """
    Incrementally ingest Tushare report_rc (brokerage earnings forecast) into MySQL.

    - Auto-creates table with a composite primary key.
    - Starts from max(report_date) present unless start_date_override provided.
    - Fetches per day with pagination; upserts into MySQL.
    """

    engine = create_engine(mysql_url, pool_recycle=3600)
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_DDL))

    # determine start/end
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y%m%d")

    start_date: Optional[str] = None
    if start_date_override and len(start_date_override) == 8:
        start_date = start_date_override
    else:
        with engine.begin() as conn:
            res = conn.execute(text(f"SELECT MAX(report_date) FROM {TABLE_NAME}"))
            row = res.fetchone()
            if row and row[0]:
                start_date = str(row[0])
    if not start_date:
        start_date = "20100101"

    # iterate days
    cur = datetime.datetime.strptime(start_date, "%Y%m%d")
    end = datetime.datetime.strptime(end_date, "%Y%m%d")
    total_rows = 0
    while cur <= end:
        day = cur.strftime("%Y%m%d")
        print(f"Fetching brokerage report {day}")
        raw = _fetch_day(day, page_limit=page_limit, sleep_secs=sleep_secs)
        if raw is not None and not raw.empty:
            df = _coerce_schema(raw)
            written = _upsert_batch(engine, df, chunksize=1000)
            total_rows += written
            print(f"{day} upserted rows ~= {written}")
        cur += datetime.timedelta(days=1)

    print(f"Completed. Total rows upserted ~= {total_rows}")


if __name__ == "__main__":
    fire.Fire(update_a_stock_brokerage_report)
