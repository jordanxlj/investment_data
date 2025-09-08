import os
import time
from typing import Optional, List, Dict, Any

import fire
import pandas as pd
import numpy as np
import tushare as ts
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
import pymysql  # noqa: F401 - required by SQLAlchemy URL


# Tushare init
ts.set_token(os.environ["TUSHARE"])  # expects env var set
pro = ts.pro_api()


TABLE_NAME = "ts_a_stock_basic"


CREATE_TABLE_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
  ts_code        VARCHAR(16)   NOT NULL,
  name           VARCHAR(64)   NOT NULL,
  area           VARCHAR(32)   NULL,
  industry       VARCHAR(32)   NULL,
  industry_code  VARCHAR(5)    NULL,
  fullname       VARCHAR(128)  NULL,
  enname         VARCHAR(128)  NULL,
  cnspell        VARCHAR(32)   NULL,
  market         VARCHAR(16)   NULL,
  exchange       VARCHAR(8)    NULL,
  curr_type      VARCHAR(8)    NULL,
  list_status    VARCHAR(8)    NULL,
  list_date      VARCHAR(8)    NULL,
  delist_date    VARCHAR(8)    NULL,
  is_hs          VARCHAR(8)    NULL,
  act_name       VARCHAR(128)  NULL,
  act_ent_type   VARCHAR(64)   NULL,
  PRIMARY KEY (ts_code),
  INDEX idx_market (market),
  INDEX idx_exchange (exchange),
  INDEX idx_list_status (list_status),
  INDEX idx_industry_code (industry_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;
"""


ALL_COLUMNS: List[str] = [
    "ts_code", "name", "area", "industry", "industry_code", "fullname", "enname",
    "cnspell", "market", "exchange", "curr_type", "list_status", "list_date",
    "delist_date", "is_hs", "act_name", "act_ent_type"
]


def _generate_industry_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Generate unique 5-digit industry codes based on industry names."""
    if df is None or df.empty or "industry" not in df.columns:
        return df

    # Create a copy to avoid modifying the original
    out = df.copy()

    # Handle None/NaN values
    out["industry"] = out["industry"].fillna("")

    # Get unique industries and generate codes
    unique_industries = out["industry"].unique()
    industry_code_map = {}

    for industry in unique_industries:
        if industry == "" or industry is None:
            industry_code_map[industry] = None
        else:
            # Generate a consistent 5-digit code using hash
            # Use only the last 5 digits to ensure it's 5 characters
            code = str(abs(hash(industry)) % 100000).zfill(5)
            industry_code_map[industry] = code

    # Apply the mapping
    out["industry_code"] = out["industry"].map(industry_code_map)

    return out


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the DataFrame schema to match our expectations."""
    # First generate industry codes
    df = _generate_industry_codes(df)

    # Ensure all expected columns exist
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Keep only known columns, in order
    out = df[ALL_COLUMNS].copy()

    # Normalize types and clean data
    if not out.empty:
        # Convert string columns to proper types
        string_cols = [
            "ts_code", "name", "area", "industry", "industry_code", "fullname", "enname",
            "cnspell", "market", "exchange", "curr_type", "list_status", "list_date",
            "delist_date", "is_hs", "act_name", "act_ent_type"
        ]

        for col in string_cols:
            if col in out.columns:
                out[col] = out[col].astype(str).replace({"nan": None, "None": None})

        # Trim long strings to prevent database errors
        out["name"] = out["name"].str.slice(0, 64)
        out["fullname"] = out["fullname"].str.slice(0, 128)
        out["enname"] = out["enname"].str.slice(0, 128)
        out["act_name"] = out["act_name"].str.slice(0, 128)
        out["area"] = out["area"].str.slice(0, 32)
        out["industry"] = out["industry"].str.slice(0, 32)
        out["cnspell"] = out["cnspell"].str.slice(0, 32)
        out["act_ent_type"] = out["act_ent_type"].str.slice(0, 64)
        out["market"] = out["market"].str.slice(0, 16)
        out["exchange"] = out["exchange"].str.slice(0, 8)
        out["curr_type"] = out["curr_type"].str.slice(0, 8)
        out["list_status"] = out["list_status"].str.slice(0, 8)
        out["list_date"] = out["list_date"].str.slice(0, 8)
        out["delist_date"] = out["delist_date"].str.slice(0, 8)
        out["is_hs"] = out["is_hs"].str.slice(0, 8)
        out["ts_code"] = out["ts_code"].str.slice(0, 16)
        out["industry_code"] = out["industry_code"].str.slice(0, 5)

        # Ensure DB NULLs: cast to object then replace NaN with None
        out = out.astype(object).where(pd.notna(out), None)
        # Extra safety for numpy.nan
        out = out.replace({np.nan: None})

    return out


def _fetch_stock_basic() -> Optional[pd.DataFrame]:
    """Fetch all stock basic information from Tushare."""
    try:
        # Exclude industry_code as it's not provided by Tushare API
        fetch_columns = [col for col in ALL_COLUMNS if col != "industry_code"]

        all_data = []

        # First fetch listed stocks (L)
        print("Fetching listed stocks...")
        df_listed = pro.stock_basic(
            exchange='',
            list_status='L',
            fields=','.join(fetch_columns)
        )
        if df_listed is not None and not df_listed.empty:
            print(f"Fetched {len(df_listed)} listed stocks")
            all_data.append(df_listed)

        # Then fetch delisted stocks (D)
        print("Fetching delisted stocks...")
        df_delisted = pro.stock_basic(
            exchange='',
            list_status='D',
            fields=','.join(fetch_columns)
        )
        if df_delisted is not None and not df_delisted.empty:
            print(f"Fetched {len(df_delisted)} delisted stocks")
            all_data.append(df_delisted)

        # Combine all data
        if not all_data:
            return pd.DataFrame(columns=fetch_columns)

        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total stocks fetched: {len(combined_df)}")
        return combined_df

    except Exception as e:
        print(f"Tushare API error: {e}")
        return None


def _upsert_batch(engine, df: pd.DataFrame, chunksize: int = 1000) -> int:
    """Upsert data in batches to MySQL database."""
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
                if c != "ts_code"  # ts_code is the primary key
            }
            ondup = stmt.on_duplicate_key_update(**update_map)
            result = conn.execute(ondup)
            total += result.rowcount or 0

    return total


def update_a_stock_basic(
    mysql_url: str = "mysql+pymysql://root:@127.0.0.1:3306/investment_data",
    force_refresh: bool = True,
) -> None:
    """
    Update stock basic information from Tushare into MySQL.

    - Auto-creates table with proper indexes.
    - Fetches all stock basic info and upserts into MySQL.
    - By default only updates if table is empty or force_refresh=True.

    Args:
        mysql_url: MySQL connection URL
        force_refresh: If True, force full refresh even if data exists
    """

    engine = create_engine(mysql_url, pool_recycle=3600)

    # Create table if it doesn't exist
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_DDL))

    # Check if we should skip update (unless force refresh)
    if not force_refresh:
        with engine.begin() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}"))
            count = result.fetchone()[0]
            if count > 0:
                print(f"Table {TABLE_NAME} already has {count} records. Use force_refresh=True to update.")
                return

    print("Fetching stock basic information from Tushare...")
    df = _fetch_stock_basic()

    if df is None or df.empty:
        print("No data fetched from Tushare")
        return

    print(f"Fetched {len(df)} stock records")

    # Process and clean data
    print("Processing data and generating industry codes...")
    df_cleaned = _coerce_schema(df)

    # Show industry code generation statistics
    unique_industries = df_cleaned["industry"].nunique()
    unique_codes = df_cleaned["industry_code"].nunique()
    print(f"Generated {unique_codes} unique industry codes for {unique_industries} industries")

    # Upsert to database
    print("Upserting data to MySQL...")
    upserted = _upsert_batch(engine, df_cleaned)

    print(f"Completed. Total records upserted: {upserted}")


if __name__ == "__main__":
    fire.Fire(update_a_stock_basic)
