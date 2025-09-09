#!/usr/bin/env python3
"""
Tushare Brokerage Report Consensus Evaluation Script

This script analyzes brokerage reports and generates consensus predictions
by aggregating ratings and forecasts from multiple analysts.

Features:
- Align brokerage report dates with financial profile periods
- Generate consensus predictions for current and next year
- Calculate sentiment indicators (bullish/bearish counts)
- Aggregate forecasts based on rating classifications
- Handle both quarterly and annual report scenarios

Usage:
    python evaluate_brokerage_report.py --eval-date 20250101
    python evaluate_brokerage_report.py --eval-date 20250101 --force-update
    python evaluate_brokerage_report.py --eval-date 20250101 --stocks "000001.SZ,000002.SZ"
"""

import os
import sys
import time
import datetime
from typing import Optional, List, Dict, Any, Tuple
from functools import wraps
from collections import defaultdict

import fire
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, and_, or_, func
from sqlalchemy.dialects.mysql import insert as mysql_insert
import pymysql  # noqa: F401 - required by SQLAlchemy URL
import tushare as ts


# Tushare init
ts.set_token(os.environ["TUSHARE"])  # expects env var set
pro = ts.pro_api()


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator: Add retry mechanism to function

    Args:
        max_retries: Maximum number of retries
        delay: Initial delay time (seconds)
        backoff: Delay multiplier
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < max_retries:
                        print(f"Function {func.__name__} call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        print(f"Waiting {current_delay:.1f} seconds before retry...")
                        time.sleep(current_delay)
                        current_delay *= backoff  # exponential backoff
                    else:
                        print(f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise last_exception

            # This line won't be executed, but for type checker
            raise last_exception

        return wrapper
    return decorator


# Rating classification mapping
RATING_MAPPING = {
    'BUY': ['BUY', 'Buy', '买入', '买进', '优于大市', '强于大市', '强力买进', '强推', '强烈推荐', '增持', '推荐', '谨慎增持', '谨慎推荐', '跑赢行业', 'OUTPERFORM', 'OVERWEIGHT', 'Overweight'],
    'HOLD': ['HOLD', 'Hold', '持有', '区间操作'],
    'NEUTRAL': ['Neutral', '中性', '无'],
    'SELL': ['SELL', 'Sell', '卖出', 'Underweight']
}


TABLE_NAME = "ts_a_stock_consensus_report"


CREATE_TABLE_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
  ts_code              VARCHAR(16)  NOT NULL,
  eval_date            VARCHAR(8)   NOT NULL,  -- 评估日期
  report_period        VARCHAR(10)  NOT NULL,  -- 报告期 (2024Q4, 2025, etc.)
  prediction_type      VARCHAR(16)  NOT NULL,  -- 预测类型: 'current_period', 'next_year'

  -- 券商报告统计信息
  total_reports        INT          NOT NULL,  -- 总报告数
  sentiment_pos        INT          NOT NULL,  -- 看多个数 (BUY + HOLD)
  sentiment_neg        INT          NOT NULL,  -- 看空个数 (NEUTRAL + SELL)
  buy_count            INT          NOT NULL,  -- BUY评级数量
  hold_count           INT          NOT NULL,  -- HOLD评级数量
  neutral_count        INT          NOT NULL,  -- NEUTRAL评级数量
  sell_count           INT          NOT NULL,  -- SELL评级数量

  -- 预测数据 (根据sentiment_pos vs sentiment_neg选择数据源)
  eps                  FLOAT NULL,   -- 每股收益预测
  pe                   FLOAT NULL,   -- 市盈率预测
  rd                   FLOAT NULL,   -- 研发费用预测
  roe                  FLOAT NULL,   -- 净资产收益率预测
  ev_ebitda           FLOAT NULL,   -- EV/EBITDA预测
  max_price           FLOAT NULL,   -- 最高价预测
  min_price           FLOAT NULL,   -- 最低价预测

  -- 数据来源标记
  data_source          VARCHAR(32)  NULL,   -- 'brokerage_consensus' or 'annual_report'
  last_updated         DATETIME     NOT NULL,

  PRIMARY KEY (ts_code, eval_date, report_period, prediction_type),
  INDEX idx_eval_date (eval_date),
  INDEX idx_ts_code (ts_code),
  INDEX idx_report_period (report_period)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;
"""


ALL_COLUMNS = [
    "ts_code", "eval_date", "report_period", "prediction_type",
    "total_reports", "sentiment_pos", "sentiment_neg",
    "buy_count", "hold_count", "neutral_count", "sell_count",
    "eps", "pe", "rd", "roe", "ev_ebitda", "max_price", "min_price",
    "data_source", "last_updated"
]


def classify_rating(rating: str) -> str:
    """
    Classify rating into BUY/HOLD/NEUTRAL/SELL categories

    Args:
        rating: Raw rating string from brokerage report

    Returns:
        Classified rating category
    """
    if not rating:
        return 'NEUTRAL'

    rating = str(rating).strip()

    for category, keywords in RATING_MAPPING.items():
        if rating in keywords:
            return category

    # Default to NEUTRAL for unrecognized ratings
    return 'NEUTRAL'


def get_date_window(eval_date: str, window_months: int = 6) -> Tuple[str, str]:
    """
    Get date window for brokerage report filtering

    Args:
        eval_date: Evaluation date in YYYYMMDD format
        window_months: Number of months to look back

    Returns:
        Tuple of (start_date, end_date) in YYYYMMDD format
    """
    eval_dt = datetime.datetime.strptime(eval_date, "%Y%m%d")
    end_dt = eval_dt
    start_dt = eval_dt - datetime.timedelta(days=window_months * 30)  # Approximate months

    start_date = start_dt.strftime("%Y%m%d")
    end_date = end_dt.strftime("%Y%m%d")

    return start_date, end_date


def get_quarter_info(eval_date: str) -> Tuple[str, str, str]:
    """
    Get quarter information for the evaluation date

    Args:
        eval_date: Evaluation date in YYYYMMDD format

    Returns:
        Tuple of (current_quarter, current_year, next_year)
    """
    eval_dt = datetime.datetime.strptime(eval_date, "%Y%m%d")
    year = eval_dt.year
    month = eval_dt.month

    # Determine current quarter
    if month <= 3:
        current_quarter = f"{year}Q1"
        current_year = f"{year}"
        next_year = f"{year + 1}"
    elif month <= 6:
        current_quarter = f"{year}Q2"
        current_year = f"{year}"
        next_year = f"{year + 1}"
    elif month <= 9:
        current_quarter = f"{year}Q3"
        current_year = f"{year}"
        next_year = f"{year + 1}"
    else:
        current_quarter = f"{year}Q4"
        current_year = f"{year}"
        next_year = f"{year + 1}"

    return current_quarter, current_year, next_year


def parse_quarter(quarter_str: str) -> Tuple[int, int]:
    """
    Parse quarter string to year and quarter number

    Args:
        quarter_str: Quarter string like '2024Q4'

    Returns:
        Tuple of (year, quarter_number)
    """
    if not quarter_str or 'Q' not in quarter_str:
        return (0, 0)

    try:
        year_str, quarter_str = quarter_str.split('Q')
        year = int(year_str)
        quarter = int(quarter_str)
        return (year, quarter)
    except (ValueError, IndexError):
        return (0, 0)


def compare_quarters(quarter1: str, quarter2: str) -> int:
    """
    Compare two quarters

    Args:
        quarter1: First quarter string
        quarter2: Second quarter string

    Returns:
        -1 if quarter1 < quarter2, 0 if equal, 1 if quarter1 > quarter2
    """
    year1, q1 = parse_quarter(quarter1)
    year2, q2 = parse_quarter(quarter2)

    if year1 < year2:
        return -1
    elif year1 > year2:
        return 1
    else:
        if q1 < q2:
            return -1
        elif q1 > q2:
            return 1
        else:
            return 0


def aggregate_forecasts(df: pd.DataFrame, sentiment_source: str) -> Dict[str, Any]:
    """
    Aggregate forecast data based on sentiment source

    Args:
        df: DataFrame with brokerage reports
        sentiment_source: 'bullish' or 'bearish'

    Returns:
        Dictionary with aggregated forecast values
    """
    if df.empty:
        return {
            'eps': None, 'pe': None, 'rd': None, 'roe': None,
            'ev_ebitda': None, 'max_price': None, 'min_price': None
        }

    forecast_fields = ['eps', 'pe', 'rd', 'roe', 'ev_ebitda', 'max_price', 'min_price']

    result = {}
    for field in forecast_fields:
        if field in df.columns:
            values = df[field].dropna()
            if not values.empty:
                result[field] = float(values.mean())
            else:
                result[field] = None
        else:
            result[field] = None

    return result


def get_brokerage_consensus(engine, ts_code: str, eval_date: str, min_quarter: str,
                          prediction_type: str) -> Optional[Dict[str, Any]]:
    """
    Get brokerage consensus for a specific stock and period

    Args:
        engine: SQLAlchemy engine
        ts_code: Stock code
        eval_date: Evaluation date
        min_quarter: Minimum quarter filter (e.g., '2024Q4')
        prediction_type: 'current_period' or 'next_year'

    Returns:
        Dictionary with consensus data or None if no data
    """
    try:
        with engine.begin() as conn:
            # Get brokerage reports within date window
            start_date, end_date = get_date_window(eval_date)

            # Get all brokerage reports first
            query = text("""
                SELECT * FROM ts_a_stock_brokerage_report
                WHERE ts_code = :ts_code
                AND report_date BETWEEN :start_date AND :end_date
                AND quarter IS NOT NULL
            """)

            df = pd.read_sql(query, conn, params={
                'ts_code': ts_code,
                'start_date': start_date,
                'end_date': end_date
            })

            if df.empty:
                return None

            # Filter by quarter if specified
            if min_quarter != 'ALL':
                df['quarter_comparison'] = df['quarter'].apply(
                    lambda q: compare_quarters(q, min_quarter) >= 0
                )
                df = df[df['quarter_comparison']]

            if df.empty:
                return None

            # Classify ratings
            df['rating_category'] = df['rating'].apply(classify_rating)

            # Count ratings
            rating_counts = df['rating_category'].value_counts()
            buy_count = rating_counts.get('BUY', 0)
            hold_count = rating_counts.get('HOLD', 0)
            neutral_count = rating_counts.get('NEUTRAL', 0)
            sell_count = rating_counts.get('SELL', 0)

            total_reports = len(df)
            sentiment_pos = buy_count + hold_count
            sentiment_neg = neutral_count + sell_count

            # Determine data source based on sentiment
            if sentiment_pos > sentiment_neg:
                # Use bullish data (BUY + HOLD)
                sentiment_df = df[df['rating_category'].isin(['BUY', 'HOLD'])]
                sentiment_source = 'bullish'
            else:
                # Use bearish data (NEUTRAL + SELL)
                sentiment_df = df[df['rating_category'].isin(['NEUTRAL', 'SELL'])]
                sentiment_source = 'bearish'

            # Aggregate forecasts
            forecasts = aggregate_forecasts(sentiment_df, sentiment_source)

            return {
                'ts_code': ts_code,
                'eval_date': eval_date,
                'report_period': min_quarter if min_quarter != 'ALL' else eval_date[:4],
                'prediction_type': prediction_type,
                'total_reports': total_reports,
                'sentiment_pos': sentiment_pos,
                'sentiment_neg': sentiment_neg,
                'buy_count': buy_count,
                'hold_count': hold_count,
                'neutral_count': neutral_count,
                'sell_count': sell_count,
                'data_source': 'brokerage_consensus',
                'last_updated': datetime.datetime.now(),
                **forecasts
            }

    except Exception as e:
        print(f"Error getting brokerage consensus for {ts_code}: {e}")
        return None


def get_annual_report_data(engine, ts_code: str, eval_date: str, report_period: str) -> Optional[Dict[str, Any]]:
    """
    Get data from annual report if available

    Args:
        engine: SQLAlchemy engine
        ts_code: Stock code
        eval_date: Evaluation date
        report_period: Report period (e.g., '2024')

    Returns:
        Dictionary with annual report data or None
    """
    try:
        with engine.begin() as conn:
            query = text("""
                SELECT * FROM ts_a_stock_financial_profile
                WHERE ts_code = :ts_code
                AND period = :report_period
                AND ann_date <= :eval_date
                ORDER BY ann_date DESC
                LIMIT 1
            """)

            df = pd.read_sql(query, conn, params={
                'ts_code': ts_code,
                'report_period': report_period,
                'eval_date': eval_date
            })

            if df.empty:
                return None

            row = df.iloc[0]

            return {
                'ts_code': ts_code,
                'eval_date': eval_date,
                'report_period': report_period,
                'prediction_type': 'current_period',
                'total_reports': 0,
                'sentiment_pos': 0,
                'sentiment_neg': 0,
                'buy_count': 0,
                'hold_count': 0,
                'neutral_count': 0,
                'sell_count': 0,
                'eps': row.get('eps'),
                'pe': None,  # Annual report doesn't have PE
                'rd': row.get('rd_exp'),
                'roe': row.get('roe_yearly'),
                'ev_ebitda': None,
                'max_price': None,
                'min_price': None,
                'data_source': 'annual_report',
                'last_updated': datetime.datetime.now()
            }

    except Exception as e:
        print(f"Error getting annual report data for {ts_code}: {e}")
        return None


def process_stock_consensus(engine, ts_code: str, eval_date: str) -> List[Dict[str, Any]]:
    """
    Process consensus data for a single stock

    Args:
        engine: SQLAlchemy engine
        ts_code: Stock code
        eval_date: Evaluation date

    Returns:
        List of consensus records
    """
    results = []
    current_quarter, current_year, next_year = get_quarter_info(eval_date)

    print(f"Processing {ts_code} for {eval_date}")

    # 1. Current period consensus (from brokerage reports)
    current_quarter_min = f"{current_year}Q4"
    current_consensus = get_brokerage_consensus(
        engine, ts_code, eval_date, current_quarter_min, 'current_period'
    )

    if current_consensus:
        results.append(current_consensus)
    else:
        # Try to get from annual report
        annual_data = get_annual_report_data(engine, ts_code, eval_date, current_year)
        if annual_data:
            results.append(annual_data)

    # 2. Next year consensus
    next_year_quarter_min = f"{next_year}Q1"
    next_year_consensus = get_brokerage_consensus(
        engine, ts_code, eval_date, next_year_quarter_min, 'next_year'
    )

    if next_year_consensus:
        results.append(next_year_consensus)

    return results


def _upsert_batch(engine, df: pd.DataFrame, chunksize: int = 1000) -> int:
    """
    Upsert consensus data in batches

    Args:
        engine: SQLAlchemy engine
        df: DataFrame with consensus data
        chunksize: Batch size for upsert

    Returns:
        Number of rows upserted
    """
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
            update_map = {
                c: getattr(stmt.inserted, c)
                for c in ALL_COLUMNS
                if c not in ("ts_code", "eval_date", "report_period", "prediction_type")
            }
            ondup = stmt.on_duplicate_key_update(**update_map)
            result = conn.execute(ondup)
            total += result.rowcount or 0

    return total


def get_stocks_list(engine, stocks: Optional[List[str]] = None) -> List[str]:
    """
    Get list of stocks to process

    Args:
        engine: SQLAlchemy engine
        stocks: Optional list of specific stocks

    Returns:
        List of stock codes
    """
    if stocks:
        return stocks

    try:
        with engine.begin() as conn:
            query = text("SELECT DISTINCT ts_code FROM ts_a_stock_brokerage_report ORDER BY ts_code")
            result = conn.execute(query)
            return [row[0] for row in result.fetchall()]
    except Exception as e:
        print(f"Error getting stocks list: {e}")
        return []


def evaluate_brokerage_report(
    mysql_url: str = "mysql+pymysql://root:@127.0.0.1:3306/investment_data",
    eval_date: str = None,
    stocks: Optional[List[str]] = None,
    force_update: bool = False,
    batch_size: int = 50
) -> None:
    """
    Main function to evaluate brokerage reports and generate consensus

    Args:
        mysql_url: MySQL connection URL
        eval_date: Evaluation date in YYYYMMDD format (default: today)
        stocks: Optional list of specific stocks to process
        force_update: Force update existing records
        batch_size: Number of stocks to process in each batch
    """
    if not eval_date:
        eval_date = datetime.datetime.now().strftime("%Y%m%d")

    print("=== Tushare Brokerage Report Consensus Evaluation ===")
    print(f"Evaluation Date: {eval_date}")
    print(f"MySQL URL: {mysql_url}")
    print(f"Force Update: {force_update}")

    engine = create_engine(mysql_url, pool_recycle=3600)

    # Create table if not exists
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_DDL))

    # Get stocks to process
    stocks_list = get_stocks_list(engine, stocks)
    if not stocks_list:
        print("No stocks found to process")
        return

    print(f"Processing {len(stocks_list)} stocks...")

    all_results = []
    processed_count = 0

    # Process stocks in batches
    for i in range(0, len(stocks_list), batch_size):
        batch_stocks = stocks_list[i:i+batch_size]
        batch_results = []

        for ts_code in batch_stocks:
            try:
                stock_results = process_stock_consensus(engine, ts_code, eval_date)
                batch_results.extend(stock_results)
                processed_count += 1

                if processed_count % 10 == 0:
                    print(f"Processed {processed_count}/{len(stocks_list)} stocks")

            except Exception as e:
                print(f"Error processing {ts_code}: {e}")
                continue

        # Save batch results
        if batch_results:
            df = pd.DataFrame(batch_results)
            if not df.empty:
                # Ensure data types
                df['last_updated'] = pd.to_datetime(df['last_updated'])
                df = df.replace({np.nan: None})

                written = _upsert_batch(engine, df)
                all_results.extend(batch_results)
                print(f"Batch {i//batch_size + 1}: upserted {written} records")

    print("
=== Evaluation Complete ===")
    print(f"Total records processed: {len(all_results)}")
    print(f"Successfully processed {processed_count} stocks")


if __name__ == "__main__":
    # Example usage:
    # python evaluate_brokerage_report.py --eval-date 20250101
    # python evaluate_brokerage_report.py --eval-date 20250101 --stocks "000001.SZ,000002.SZ"
    # python evaluate_brokerage_report.py --eval-date 20250101 --force-update
    fire.Fire(evaluate_brokerage_report)
