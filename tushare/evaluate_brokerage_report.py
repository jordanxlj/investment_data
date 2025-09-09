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
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from functools import wraps, lru_cache
from collections import defaultdict
import concurrent.futures

import fire
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, and_, or_, func, MetaData, Table
from sqlalchemy.dialects.mysql import insert as mysql_insert
import pymysql  # noqa: F401 - required by SQLAlchemy URL
import tushare as ts


# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluate_brokerage_report.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


# Tushare init with validation
TUSHARE_TOKEN = os.environ.get("TUSHARE")
if not TUSHARE_TOKEN:
    logger.error("TUSHARE environment variable not set")
    sys.exit(1)

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


# Rating classification mapping
RATING_MAPPING = {
    'BUY': ['BUY', 'Buy', '买入', '买进', '优于大市', '强于大市', '强力买进', '强推', '强烈推荐', '增持', '推荐', '谨慎增持', '谨慎推荐', '跑赢行业', 'OUTPERFORM', 'OVERWEIGHT', 'Overweight'],
    'HOLD': ['HOLD', 'Hold', '持有', '区间操作'],
    'NEUTRAL': ['Neutral', '中性', '无'],
    'SELL': ['SELL', 'Sell', '卖出', 'Underweight']
}

# Report type weight mapping
REPORT_TYPE_WEIGHTS = {
    '深度': 5.0,      # Depth analysis - highest weight
    'depth': 5.0,
    '调研': 4.0,      # Field research - high weight
    'research': 4.0,
    'field': 4.0,
    '点评': 3.0,      # Commentary - medium weight
    'commentary': 3.0,
    'comment': 3.0,
    '一般': 2.0,      # General - low weight
    'general': 2.0,
    '非各股': 1.0,    # Non-stock specific - lowest weight
    'non-stock': 1.0,
    'industry': 1.0,
    'strategy': 1.0
}

DEFAULT_REPORT_WEIGHT = 2.0  # Default weight for unrecognized types

TABLE_NAME = "ts_a_stock_consensus_report"

CREATE_TABLE_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
  ts_code              VARCHAR(16)  NOT NULL,
  eval_date            VARCHAR(8)   NOT NULL,  -- 评估日期
  report_period        VARCHAR(10)  NOT NULL,  -- 报告期 (2024Q4, 2025, etc.)

  -- 券商报告统计信息
  total_reports        INT          NOT NULL,  -- 总报告数
  sentiment_pos        INT          NOT NULL,  -- 看多个数 (BUY + HOLD)
  sentiment_neg        INT          NOT NULL,  -- 看空个数 (NEUTRAL + SELL)
  buy_count            INT          NOT NULL,  -- BUY评级数量
  hold_count           INT          NOT NULL,  -- HOLD评级数量
  neutral_count        INT          NOT NULL,  -- NEUTRAL评级数量
  sell_count           INT          NOT NULL,  -- SELL评级数量

  -- 研报类型权重统计
  depth_reports        INT          NOT NULL DEFAULT 0,  -- 深度研报数量
  research_reports     INT          NOT NULL DEFAULT 0,  -- 调研研报数量
  commentary_reports   INT          NOT NULL DEFAULT 0,  -- 点评研报数量
  general_reports      INT          NOT NULL DEFAULT 0,  -- 一般研报数量
  other_reports        INT          NOT NULL DEFAULT 0,  -- 其他研报数量
  avg_report_weight    FLOAT        NULL,               -- 平均研报权重

  -- 当前周期预测数据 (根据sentiment_pos vs sentiment_neg选择数据源)
  eps                  FLOAT NULL,   -- 每股收益预测
  pe                   FLOAT NULL,   -- 市盈率预测
  rd                   FLOAT NULL,   -- 研发费用预测
  roe                  FLOAT NULL,   -- 净资产收益率预测
  ev_ebitda           FLOAT NULL,   -- EV/EBITDA预测
  max_price           FLOAT NULL,   -- 最高价预测
  min_price           FLOAT NULL,   -- 最低价预测

  -- 下一年度预测数据
  next_year_eps        FLOAT NULL,   -- 下一年每股收益预测
  next_year_pe         FLOAT NULL,   -- 下一年市盈率预测
  next_year_roe        FLOAT NULL,   -- 下一年净资产收益率预测
  next_year_ev_ebitda  FLOAT NULL,   -- 下一年EV/EBITDA预测

  -- 下一年度统计信息
  next_year_reports    INT          NOT NULL DEFAULT 0,  -- 下一年报告数
  next_year_avg_weight FLOAT        NULL,               -- 下一年平均研报权重

  -- 数据来源标记
  data_source          VARCHAR(32)  NULL,   -- 'brokerage_consensus' or 'annual_report'
  last_updated         DATETIME     NOT NULL,

  PRIMARY KEY (ts_code, eval_date, report_period),
  INDEX idx_eval_date (eval_date),
  INDEX idx_ts_code (ts_code),
  INDEX idx_report_period (report_period)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;
"""


ALL_COLUMNS = [
    "ts_code", "eval_date", "report_period",
    "total_reports", "sentiment_pos", "sentiment_neg",
    "buy_count", "hold_count", "neutral_count", "sell_count",
    "depth_reports", "research_reports", "commentary_reports",
    "general_reports", "other_reports", "avg_report_weight",
    "eps", "pe", "rd", "roe", "ev_ebitda", "max_price", "min_price",
    "next_year_eps", "next_year_pe", "next_year_roe", "next_year_ev_ebitda",
    "next_year_reports", "next_year_avg_weight",
    "data_source", "last_updated"
]


def get_report_weight(report_type: str) -> float:
    """
    Get weight for a report type

    Args:
        report_type: Report type string

    Returns:
        Weight value (1.0 to 5.0)
    """
    if not report_type:
        logger.debug(f"Report type is empty, using default weight: {DEFAULT_REPORT_WEIGHT}")
        return DEFAULT_REPORT_WEIGHT

    report_type_lower = str(report_type).strip().lower()
    logger.debug(f"Processing report type: '{report_type}' -> '{report_type_lower}'")

    # Direct match first
    if report_type in REPORT_TYPE_WEIGHTS:
        weight = REPORT_TYPE_WEIGHTS[report_type]
        logger.debug(f"Direct match found for '{report_type}': weight = {weight}")
        return weight

    # Partial match
    for key, weight in REPORT_TYPE_WEIGHTS.items():
        if key.lower() in report_type_lower:
            logger.debug(f"Partial match found for '{report_type}' with key '{key}': weight = {weight}")
            return weight

    logger.debug(f"No match found for '{report_type}', using default weight: {DEFAULT_REPORT_WEIGHT}")
    return DEFAULT_REPORT_WEIGHT


def categorize_report_type(report_type: str) -> str:
    """
    Categorize report type into main categories

    Args:
        report_type: Report type string

    Returns:
        Category string ('depth', 'research', 'commentary', 'general', 'other')
    """
    if not report_type:
        logger.debug("Report type is empty, categorizing as 'other'")
        return 'other'

    report_type_lower = str(report_type).strip().lower()
    logger.debug(f"Categorizing report type: '{report_type}' -> '{report_type_lower}'")

    # Check for depth reports
    if any(keyword in report_type_lower for keyword in ['深度', 'depth', 'comprehensive', 'detailed']):
        logger.debug(f"Categorized '{report_type}' as 'depth'")
        return 'depth'

    # Check for research reports
    if any(keyword in report_type_lower for keyword in ['调研', 'research', 'field', 'visit', 'survey']):
        logger.debug(f"Categorized '{report_type}' as 'research'")
        return 'research'

    # Check for commentary reports
    if any(keyword in report_type_lower for keyword in ['点评', 'commentary', 'comment', 'analysis', 'review']):
        logger.debug(f"Categorized '{report_type}' as 'commentary'")
        return 'commentary'

    # Check for general reports
    if any(keyword in report_type_lower for keyword in ['一般', 'general', 'regular', 'standard']):
        logger.debug(f"Categorized '{report_type}' as 'general'")
        return 'general'

    # Check for non-stock specific reports
    if any(keyword in report_type_lower for keyword in ['非各股', 'non-stock', 'industry', 'strategy', 'sector']):
        logger.debug(f"Categorized '{report_type}' as 'other'")
        return 'other'

    logger.debug(f"No category match for '{report_type}', defaulting to 'other'")
    return 'other'


def classify_rating(rating: str) -> str:
    """
    Classify rating into BUY/HOLD/NEUTRAL/SELL categories

    Args:
        rating: Raw rating string from brokerage report

    Returns:
        Classified rating category
    """
    if not rating:
        logger.debug("Rating is empty, defaulting to 'NEUTRAL'")
        return 'NEUTRAL'

    rating = str(rating).strip()
    logger.debug(f"Classifying rating: '{rating}'")

    for category, keywords in RATING_MAPPING.items():
        if rating in keywords:
            logger.debug(f"Rating '{rating}' classified as '{category}'")
            return category

    # Default to NEUTRAL for unrecognized ratings
    logger.debug(f"Rating '{rating}' not found in mapping, defaulting to 'NEUTRAL'")
    return 'NEUTRAL'


def get_trade_cal(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get trading calendar from Tushare API

    Args:
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format

    Returns:
        DataFrame with trading dates
    """
    logger.debug(f"Fetching trading calendar from {start_date} to {end_date}")
    try:
        df = pro.trade_cal(exchange='SSE', is_open='1',
                          start_date=start_date,
                          end_date=end_date,
                          fields='cal_date')
        logger.debug(f"Trading calendar fetched successfully: {len(df)} trading days found")
        if len(df) > 0:
            logger.debug(f"First trading date: {df['cal_date'].min()}, Last trading date: {df['cal_date'].max()}")
        return df
    except Exception as e:
        logger.error(f"Error getting trade calendar: {e}")
        logger.debug("Returning empty DataFrame as fallback")
        return pd.DataFrame()


def get_date_window(eval_date: str, window_months: int = 6) -> Tuple[str, str]:
    """
    Get date window for brokerage report filtering

    Args:
        eval_date: Evaluation date in YYYYMMDD format
        window_months: Number of months to look back

    Returns:
        Tuple of (start_date, end_date) in YYYYMMDD format
    """
    # Ensure eval_date is a string
    eval_date = str(eval_date)
    logger.debug(f"Calculating date window for eval_date: {eval_date}, window_months: {window_months}")
    eval_dt = datetime.datetime.strptime(eval_date, "%Y%m%d")
    end_dt = eval_dt
    start_dt = eval_dt - datetime.timedelta(days=window_months * 30)  # Approximate months

    start_date = start_dt.strftime("%Y%m%d")
    end_date = end_dt.strftime("%Y%m%d")

    logger.debug(f"Date window calculated: {start_date} to {end_date}")
    return start_date, end_date


def get_fiscal_period_info(eval_date: str) -> Dict[str, Any]:
    """
    Get comprehensive fiscal period information for the evaluation date

    Args:
        eval_date: Evaluation date in YYYYMMDD format

    Returns:
        Dictionary with fiscal period information
    """
    # Ensure eval_date is a string
    eval_date = str(eval_date)
    logger.debug(f"Getting fiscal period info for eval_date: {eval_date}")
    eval_dt = datetime.datetime.strptime(eval_date, "%Y%m%d")
    year = eval_dt.year
    month = eval_dt.month

    logger.debug(f"Parsed date: year={year}, month={month}")

    # Determine current fiscal periods
    if month <= 3:
        current_quarter = f"{year}Q1"
        current_year = f"{year}"
        next_year = f"{year + 1}"
        # For Q1 evaluation, we look at previous year Q4 data
        current_fiscal_year = f"{year - 1}"
        current_fiscal_period = f"{year - 1}1231"  # Previous year end
        logger.debug("Fiscal period: Q1 - looking at previous year Q4 data")
    elif month <= 6:
        current_quarter = f"{year}Q2"
        current_year = f"{year}"
        next_year = f"{year + 1}"
        current_fiscal_year = f"{year}"
        current_fiscal_period = f"{year}0331"  # Q1 end
        logger.debug("Fiscal period: Q2 - looking at Q1 data")
    elif month <= 9:
        current_quarter = f"{year}Q3"
        current_year = f"{year}"
        next_year = f"{year + 1}"
        current_fiscal_year = f"{year}"
        current_fiscal_period = f"{year}0630"  # H1 end
        logger.debug("Fiscal period: Q3 - looking at H1 data")
    else:
        current_quarter = f"{year}Q4"
        current_year = f"{year}"
        next_year = f"{year + 1}"
        current_fiscal_year = f"{year}"
        current_fiscal_period = f"{year}0930"  # Q3 end
        logger.debug("Fiscal period: Q4 - looking at Q3 data")

    fiscal_info = {
        'eval_date': eval_date,
        'eval_datetime': eval_dt,
        'current_quarter': current_quarter,
        'current_year': current_year,
        'next_year': next_year,
        'current_fiscal_year': current_fiscal_year,
        'current_fiscal_period': current_fiscal_period,
        'next_fiscal_year': f"{year + 1}",
        'next_fiscal_period': f"{year + 1}0331" if month <= 3 else f"{year + 1}1231"
    }

    logger.debug(f"Fiscal info: current_quarter={current_quarter}, current_fiscal_year={current_fiscal_year}, "
                f"next_year={next_year}")
    return fiscal_info


def get_quarter_info(eval_date: str) -> Tuple[str, str, str]:
    """
    Get quarter information for the evaluation date (backward compatibility)

    Args:
        eval_date: Evaluation date in YYYYMMDD format

    Returns:
        Tuple of (current_quarter, current_year, next_year)
    """
    info = get_fiscal_period_info(eval_date)
    return info['current_quarter'], info['current_year'], info['next_year']


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
    Aggregate forecast data based on sentiment source with robust handling and weighting

    Args:
        df: DataFrame with brokerage reports
        sentiment_source: 'bullish' or 'bearish'

    Returns:
        Dictionary with aggregated forecast values
    """
    logger.debug(f"Aggregating forecasts for {len(df)} reports with sentiment: {sentiment_source}")

    if df.empty:
        logger.debug("Empty DataFrame provided for aggregation")
        return {
            'eps': None, 'pe': None, 'rd': None, 'roe': None,
            'ev_ebitda': None, 'max_price': None, 'min_price': None
        }

    # Add weights to the dataframe
    df = df.copy()
    df['report_weight'] = df['report_type'].apply(get_report_weight)
    logger.debug(f"Added report weights: min={df['report_weight'].min():.1f}, max={df['report_weight'].max():.1f}, "
                f"avg={df['report_weight'].mean():.1f}")

    forecast_fields = ['eps', 'pe', 'rd', 'roe', 'ev_ebitda', 'max_price', 'min_price']

    result = {}
    for field in forecast_fields:
        logger.debug(f"Processing field: {field}")
        if field in df.columns:
            # Get values and weights for valid data points
            field_data = df[[field, 'report_weight']].dropna()

            if field_data.empty:
                result[field] = None
                logger.debug(f"{field}: no valid values after filtering")
                continue

            values = field_data[field]
            weights = field_data['report_weight']
            logger.debug(f"{field}: {len(values)} values before outlier filtering")

            # Remove extreme outliers (beyond 3 standard deviations)
            if len(values) > 2:
                mean_val = values.mean()
                std_val = values.std()
                logger.debug(f"{field}: raw mean={mean_val:.2f}, std={std_val:.2f}")
                if std_val > 0:
                    valid_mask = (values >= mean_val - 3 * std_val) & (values <= mean_val + 3 * std_val)
                    values = values[valid_mask]
                    weights = weights[valid_mask]
                    logger.debug(f"{field}: {len(values)} values after outlier filtering")

            # Remove unrealistic values based on field type
            if field == 'eps':
                valid_mask = (values >= -50) & (values <= 50)  # EPS between -50 and 50
                logger.debug(f"{field}: EPS range filter applied")
            elif field in ['pe', 'ev_ebitda']:
                valid_mask = (values > 0) & (values <= 500)  # Positive, reasonable multiples
                logger.debug(f"{field}: positive multiples filter applied")
            elif field == 'roe':
                valid_mask = (values >= -200) & (values <= 200)  # ROE between -200% and 200%
                logger.debug(f"{field}: ROE range filter applied")
            elif field in ['max_price', 'min_price']:
                valid_mask = (values > 0) & (values <= 10000)  # Positive, reasonable prices
                logger.debug(f"{field}: price range filter applied")
            else:
                valid_mask = pd.Series(True, index=values.index)

            values = values[valid_mask]
            weights = weights[valid_mask]
            logger.debug(f"{field}: {len(values)} values after all filters")

            if not values.empty:
                # Use weighted median for robustness
                if len(values) == 1:
                    result[field] = float(values.iloc[0])
                    logger.debug(f"{field}: single value used = {result[field]:.2f}")
                else:
                    # Calculate weighted median
                    result[field] = float(weighted_median(values.values, weights.values))
                    logger.debug(f"{field}: {len(values)} values, weights {weights.min():.1f}-{weights.max():.1f}, "
                               f"{sentiment_source} weighted median = {result[field]:.2f}")
            else:
                result[field] = None
                logger.debug(f"{field}: no valid values after filtering")
        else:
            result[field] = None
            logger.debug(f"{field}: column not found in DataFrame")

    logger.debug(f"Aggregation completed for {sentiment_source}: {len([v for v in result.values() if v is not None])} fields with values")
    return result


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate weighted median

    Args:
        values: Array of values
        weights: Array of weights corresponding to values

    Returns:
        Weighted median value
    """
    if len(values) != len(weights):
        raise ValueError("Values and weights must have the same length")

    if len(values) == 0:
        raise ValueError("Cannot calculate median of empty array")

    # Sort values and corresponding weights
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Calculate cumulative weights
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]

    # Find the weighted median
    median_weight = total_weight / 2
    median_index = np.searchsorted(cum_weights, median_weight, side='right')

    if median_index == 0:
        return sorted_values[0]
    elif median_index >= len(sorted_values):
        return sorted_values[-1]
    else:
        return sorted_values[median_index - 1]


@lru_cache(maxsize=1000)
def get_brokerage_consensus(engine, ts_code: str, eval_date: str, min_quarter: str) -> Optional[Dict[str, Any]]:
    """
    Get brokerage consensus for a specific stock and period with enhanced filtering

    Args:
        engine: SQLAlchemy engine
        ts_code: Stock code
        eval_date: Evaluation date
        min_quarter: Minimum quarter filter (e.g., '2024Q4')

    Returns:
        Dictionary with consensus data including current and next year forecasts or None if no data
    """
    logger.debug(f"Getting brokerage consensus for {ts_code}, eval_date: {eval_date}, min_quarter: {min_quarter}")
    try:
        with engine.begin() as conn:
            # Get brokerage reports within date window
            start_date, end_date = get_date_window(eval_date)

            # Enhanced query with age filtering (reports not older than 1 year)
            max_age_date = (datetime.datetime.strptime(eval_date, "%Y%m%d") -
                           datetime.timedelta(days=365)).strftime("%Y%m%d")

            logger.debug(f"Querying reports for {ts_code} from {start_date} to {end_date}, max age: {max_age_date}")

            # Get all brokerage reports first
            query = text("""
                SELECT * FROM ts_a_stock_brokerage_report
                WHERE ts_code = :ts_code
                AND report_date BETWEEN :start_date AND :end_date
                AND report_date >= :max_age_date
                AND quarter IS NOT NULL
                AND rating IS NOT NULL
                AND report_type IS NOT NULL
            """)

            df = pd.read_sql(query, conn, params={
                'ts_code': ts_code,
                'start_date': start_date,
                'end_date': end_date,
                'max_age_date': max_age_date
            })

            if df.empty:
                logger.debug(f"No brokerage reports found for {ts_code} in date range")
                return None

            # Filter by quarter if specified
            if min_quarter != 'ALL':
                df['quarter_comparison'] = df['quarter'].apply(
                    lambda q: compare_quarters(q, min_quarter) >= 0
                )
                df = df[df['quarter_comparison']]
                logger.debug(f"After quarter filtering (>= {min_quarter}): {len(df)} reports remain")

            if df.empty:
                logger.debug(f"No reports after quarter filtering for {ts_code}")
                return None

            logger.debug(f"Found {len(df)} reports for {ts_code} after filtering")
            logger.debug(f"Report date range: {df['report_date'].min()} to {df['report_date'].max()}")

            # Classify ratings and report types
            df['rating_category'] = df['rating'].apply(classify_rating)
            df['report_weight'] = df['report_type'].apply(get_report_weight)
            df['report_type_category'] = df['report_type'].apply(categorize_report_type)

            logger.debug(f"Classification completed for {len(df)} reports")

            # Count ratings
            rating_counts = df['rating_category'].value_counts()
            buy_count = rating_counts.get('BUY', 0)
            hold_count = rating_counts.get('HOLD', 0)
            neutral_count = rating_counts.get('NEUTRAL', 0)
            sell_count = rating_counts.get('SELL', 0)

            # Count report types
            report_type_counts = df['report_type_category'].value_counts()
            depth_reports = report_type_counts.get('depth', 0)
            research_reports = report_type_counts.get('research', 0)
            commentary_reports = report_type_counts.get('commentary', 0)
            general_reports = report_type_counts.get('general', 0)
            other_reports = report_type_counts.get('other', 0)

            total_reports = len(df)
            sentiment_pos = buy_count + hold_count
            sentiment_neg = neutral_count + sell_count
            avg_report_weight = df['report_weight'].mean() if not df.empty else 0.0

            logger.debug(f"{ts_code}: BUY={buy_count}, HOLD={hold_count}, NEUTRAL={neutral_count}, SELL={sell_count}")
            logger.debug(f"{ts_code}: Sentiment POS={sentiment_pos}, NEG={sentiment_neg}")
            logger.debug(f"{ts_code}: Report types - Depth:{depth_reports}, Research:{research_reports}, "
                        f"Commentary:{commentary_reports}, General:{general_reports}, Other:{other_reports}")
            logger.debug(f"{ts_code}: Avg weight: {avg_report_weight:.2f}")

            # Determine data source based on sentiment
            if sentiment_pos > sentiment_neg:
                # Use bullish data (BUY + HOLD)
                sentiment_df = df[df['rating_category'].isin(['BUY', 'HOLD'])]
                sentiment_source = 'bullish'
            elif sentiment_neg > sentiment_pos:
                # Use bearish data (NEUTRAL + SELL)
                sentiment_df = df[df['rating_category'].isin(['NEUTRAL', 'SELL'])]
                sentiment_source = 'bearish'
            else:
                # Tie - use all data
                sentiment_df = df
                sentiment_source = 'neutral'

            logger.debug(f"{ts_code}: Using {sentiment_source} data with {len(sentiment_df)} reports")
            logger.debug(f"{ts_code}: Sentiment breakdown - POS: {sentiment_pos}, NEG: {sentiment_neg}, TIE: {sentiment_pos == sentiment_neg}")

            # Aggregate forecasts
            forecasts = aggregate_forecasts(sentiment_df, sentiment_source)

            return {
                'ts_code': ts_code,
                'eval_date': eval_date,
                'report_period': min_quarter if min_quarter != 'ALL' else eval_date[:4],
                'total_reports': total_reports,
                'sentiment_pos': sentiment_pos,
                'sentiment_neg': sentiment_neg,
                'buy_count': buy_count,
                'hold_count': hold_count,
                'neutral_count': neutral_count,
                'sell_count': sell_count,
                'depth_reports': depth_reports,
                'research_reports': research_reports,
                'commentary_reports': commentary_reports,
                'general_reports': general_reports,
                'other_reports': other_reports,
                'avg_report_weight': avg_report_weight,
                # 当前周期预测数据
                'eps': forecasts.get('eps'),
                'pe': forecasts.get('pe'),
                'rd': forecasts.get('rd'),
                'roe': forecasts.get('roe'),
                'ev_ebitda': forecasts.get('ev_ebitda'),
                'max_price': forecasts.get('max_price'),
                'min_price': forecasts.get('min_price'),
                # 下一年预测数据（暂时为空，后续函数会填充）
                'next_year_eps': None,
                'next_year_pe': None,
                'next_year_roe': None,
                'next_year_ev_ebitda': None,
                'next_year_reports': 0,
                'next_year_avg_weight': 0.0,
                'data_source': 'brokerage_consensus',
                'last_updated': datetime.datetime.now()
            }

    except Exception as e:
        logger.error(f"Error getting brokerage consensus for {ts_code}: {e}")
        return None


def get_next_year_consensus(engine, ts_code: str, eval_date: str, next_year: str) -> Optional[Dict[str, Any]]:
    """
    Get next year brokerage consensus for a specific stock

    Args:
        engine: SQLAlchemy engine
        ts_code: Stock code
        eval_date: Evaluation date
        next_year: Next year (e.g., '2025')

    Returns:
        Dictionary with next year consensus data or None if no data
    """
    try:
        with engine.begin() as conn:
            # Get brokerage reports within date window
            start_date, end_date = get_date_window(eval_date)

            # Get reports that predict the next year
            query = text("""
                SELECT * FROM ts_a_stock_brokerage_report
                WHERE ts_code = :ts_code
                AND report_date BETWEEN :start_date AND :end_date
                AND report_date >= :max_age_date
                AND quarter IS NOT NULL
                AND rating IS NOT NULL
                AND report_type IS NOT NULL
                AND quarter LIKE :next_year_pattern
            """)

            df = pd.read_sql(query, conn, params={
                'ts_code': ts_code,
                'start_date': start_date,
                'end_date': end_date,
                'max_age_date': (datetime.datetime.strptime(eval_date, "%Y%m%d") -
                               datetime.timedelta(days=365)).strftime("%Y%m%d"),
                'next_year_pattern': f"{next_year}Q%"
            })

            if df.empty:
                logger.debug(f"No next year reports found for {ts_code} in {next_year}")
                return None

            logger.debug(f"Found {len(df)} next year reports for {ts_code}")

            # Add weights for next year reports (情感指标对下一年预测不适用)
            df['report_weight'] = df['report_type'].apply(get_report_weight)

            total_reports = len(df)
            avg_report_weight = df['report_weight'].mean() if not df.empty else 0.0

            # 对于下一年预测，使用所有数据（不区分情感）
            sentiment_df = df
            sentiment_source = 'next_year'

            # Aggregate forecasts for next year (focus on key metrics)
            next_year_forecasts = aggregate_forecasts(sentiment_df, sentiment_source)

            return {
                'total_reports': total_reports,
                'avg_report_weight': avg_report_weight,
                'eps': next_year_forecasts.get('eps'),
                'pe': next_year_forecasts.get('pe'),
                'roe': next_year_forecasts.get('roe'),
                'ev_ebitda': next_year_forecasts.get('ev_ebitda')
            }

    except Exception as e:
        logger.error(f"Error getting next year consensus for {ts_code}: {e}")
        return None


def get_annual_report_data(engine, ts_code: str, eval_date: str, report_period: str) -> Optional[Dict[str, Any]]:
    """
    Get data from annual report and fundamental data if available

    Args:
        engine: SQLAlchemy engine
        ts_code: Stock code
        eval_date: Evaluation date
        report_period: Report period (e.g., '2024')

    Returns:
        Dictionary with annual report data or None
    """
    logger.debug(f"Getting annual report data for {ts_code}, period: {report_period}, eval_date: {eval_date}")
    try:
        with engine.begin() as conn:
            # Get annual report data from financial profile
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
                logger.debug(f"No annual report found for {ts_code} in period {report_period}")
                return None

            row = df.iloc[0]
            logger.debug(f"Found annual report for {ts_code}: ann_date={row.get('ann_date')}, period={row.get('period')}")

            # Get PE and dividend ratio from fundamental data
            fundamental_query = text("""
                SELECT pe, dv_ratio FROM ts_a_stock_fundamental
                WHERE ts_code = :ts_code
                AND trade_date <= :eval_date
                ORDER BY trade_date DESC
                LIMIT 1
            """)

            pe_value = None
            dv_ratio_value = None

            try:
                fundamental_df = pd.read_sql(fundamental_query, conn, params={
                    'ts_code': ts_code,
                    'eval_date': eval_date
                })

                if not fundamental_df.empty:
                    pe_value = fundamental_df.iloc[0].get('pe')
                    dv_ratio_value = fundamental_df.iloc[0].get('dv_ratio')
                    logger.debug(f"Fundamental data for {ts_code}: PE={pe_value}, DV_RATIO={dv_ratio_value}")
                else:
                    logger.debug(f"No fundamental data found for {ts_code}")
            except Exception as e:
                logger.debug(f"Could not get fundamental data for {ts_code}: {e}")

            return {
                'ts_code': ts_code,
                'eval_date': eval_date,
                'report_period': report_period,
                'total_reports': None,
                'sentiment_pos': None,
                'sentiment_neg': None,
                'buy_count': None,
                'hold_count': None,
                'neutral_count': None,
                'sell_count': None,
                'depth_reports': None,
                'research_reports': None,
                'commentary_reports': None,
                'general_reports': None,
                'other_reports': None,
                'avg_report_weight': None,
                # 当前周期预测数据
                'eps': row.get('eps'),
                'pe': pe_value,  # From fundamental data
                'rd': dv_ratio_value,  # Dividend ratio from fundamental data (not rd_exp)
                'roe': row.get('roe_waa'),  # Use weighted average ROE
                'ev_ebitda': None,
                'max_price': None,
                'min_price': None,
                # 下一年预测数据
                'next_year_eps': None,
                'next_year_pe': None,
                'next_year_roe': None,
                'next_year_ev_ebitda': None,
                'next_year_reports': None,
                'next_year_avg_weight': None,
                'data_source': 'annual_report',
                'last_updated': datetime.datetime.now()
            }

    except Exception as e:
        logger.error(f"Error getting annual report data for {ts_code}: {e}")
        return None


def process_stock_consensus(engine, ts_code: str, eval_date: str) -> Optional[Dict[str, Any]]:
    """
    Process consensus data for a single stock with enhanced fiscal period handling

    Args:
        engine: SQLAlchemy engine
        ts_code: Stock code
        eval_date: Evaluation date

    Returns:
        Dictionary with combined current and next year consensus data or None
    """
    logger.debug(f"Starting process_stock_consensus for {ts_code} on {eval_date}")
    fiscal_info = get_fiscal_period_info(eval_date)

    logger.info(f"Processing {ts_code} for {eval_date} (fiscal year: {fiscal_info['current_fiscal_year']})")
    logger.debug(f"Fiscal details: current_period={fiscal_info['current_fiscal_period']}, "
                f"next_year={fiscal_info['next_fiscal_year']}")

    # 1. Get current period consensus (from brokerage reports)
    logger.debug(f"Step 1: Getting current period consensus for {ts_code}")
    current_consensus = get_brokerage_consensus(
        engine, ts_code, eval_date, fiscal_info['current_fiscal_year']
    )

    if not current_consensus:
        # Try to get from annual report if we have fiscal year data
        logger.debug(f"No brokerage consensus found for {ts_code}, trying annual report data")
        annual_data = get_annual_report_data(engine, ts_code, eval_date, fiscal_info['current_fiscal_period'])
        if annual_data:
            current_consensus = annual_data
            logger.info(f"{ts_code}: Using annual report data for current period")
        else:
            logger.warning(f"{ts_code}: No data found for current period")
            return None

    # 2. Get next year consensus
    logger.debug(f"Step 2: Getting next year consensus for {ts_code}")
    next_year_data = get_next_year_consensus(
        engine, ts_code, eval_date, fiscal_info['next_fiscal_year']
    )

    if next_year_data:
        logger.info(f"{ts_code}: Found {next_year_data['total_reports']} reports for next year")
    else:
        logger.debug(f"{ts_code}: No next year data found")

    # 3. Combine current and next year data into single record
    logger.debug(f"Step 3: Combining current and next year data for {ts_code}")
    combined_result = current_consensus.copy()

    if next_year_data:
        combined_result.update({
            'next_year_eps': next_year_data['eps'],
            'next_year_pe': next_year_data['pe'],
            'next_year_roe': next_year_data['roe'],
            'next_year_ev_ebitda': next_year_data['ev_ebitda'],
            'next_year_reports': next_year_data['total_reports'],
            'next_year_avg_weight': next_year_data['avg_report_weight']
        })
        logger.debug(f"{ts_code}: Added next year data to combined result")

    logger.info(f"{ts_code}: Processed consensus with {combined_result['total_reports']} current + "
               f"{combined_result.get('next_year_reports', 0)} next year reports")

    return combined_result


def _upsert_batch(engine, df: pd.DataFrame, chunksize: int = 1000) -> int:
    """
    Upsert consensus data in batches with optimized performance

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
    meta = MetaData()
    table = Table(TABLE_NAME, meta, autoload_with=engine)

    # Prepare data for bulk operations
    rows = df.to_dict(orient="records")

    with engine.begin() as conn:
        for i in range(0, len(rows), chunksize):
            batch = rows[i:i+chunksize]

            # Use bulk insert with on duplicate key update
            stmt = mysql_insert(table).values(batch)
            update_map = {
                c: getattr(stmt.inserted, c)
                for c in ALL_COLUMNS
                if c not in ("ts_code", "eval_date", "report_period")
            }
            ondup = stmt.on_duplicate_key_update(**update_map)

            try:
                result = conn.execute(ondup)
                batch_count = result.rowcount or 0
                total += batch_count
                logger.debug(f"Upserted batch {i//chunksize + 1}: {batch_count} rows")
            except Exception as e:
                logger.error(f"Error upserting batch {i//chunksize + 1}: {e}")
                # Continue with next batch rather than failing completely
                continue

    return total


def get_stocks_list(engine, stocks: Optional[List[str]] = None) -> List[str]:
    """
    Get list of active stocks to process from ts_a_stock_basic table

    Args:
        engine: SQLAlchemy engine
        stocks: Optional list of specific stocks

    Returns:
        List of active stock codes
    """
    if stocks:
        logger.debug(f"Using provided stock list: {len(stocks)} stocks")
        return stocks

    logger.debug("Getting active stocks from database")
    try:
        with engine.begin() as conn:
            # Query active stocks from ts_a_stock_basic table
            # list_status = 'L' means listed, and delist_date is null or in the future
            query = text("""
                SELECT ts_code FROM ts_a_stock_basic
                WHERE list_status = 'L'
                AND (delist_date IS NULL OR delist_date = '' OR delist_date > :current_date)
                ORDER BY ts_code
            """)

            current_date = datetime.datetime.now().strftime("%Y%m%d")
            logger.debug(f"Querying active stocks with current_date={current_date}")
            result = conn.execute(query, {'current_date': current_date})
            stock_codes = [row[0] for row in result.fetchall()]

            logger.info(f"Found {len(stock_codes)} active stocks in ts_a_stock_basic")
            if len(stock_codes) > 0:
                logger.debug(f"Sample stocks: {stock_codes[:5]}...")
            return stock_codes

    except Exception as e:
        logger.error(f"Error getting stocks list from ts_a_stock_basic: {e}")
        # Fallback to brokerage report table if basic table query fails
        try:
            logger.info("Falling back to ts_a_stock_brokerage_report for stock list")
            with engine.begin() as conn:
                query = text("SELECT DISTINCT ts_code FROM ts_a_stock_brokerage_report ORDER BY ts_code")
                result = conn.execute(query)
                fallback_stocks = [row[0] for row in result.fetchall()]
                logger.info(f"Fallback query found {len(fallback_stocks)} stocks")
                return fallback_stocks
        except Exception as e2:
            logger.error(f"Fallback query also failed: {e2}")
            return []


def evaluate_brokerage_report(
    mysql_url: str = None,
    start_date: str = None,
    end_date: str = None,
    stocks: Optional[List[str]] = None,
    force_update: bool = False,
    batch_size: int = 50,
    max_workers: int = 4,
    dry_run: bool = False
) -> None:
    """
    Main function to evaluate brokerage reports and generate consensus for date range

    Args:
        mysql_url: MySQL connection URL (default: from env)
        start_date: Start date in YYYYMMDD format (default: today)
        end_date: End date in YYYYMMDD format (default: today)
        stocks: Optional list of specific stocks to process
        force_update: Force update existing records
        batch_size: Number of stocks to process in each batch
        max_workers: Maximum number of parallel workers
        dry_run: If True, only show what would be processed without writing to DB
    """
    # Parameter validation and setup
    if mysql_url is None:
        mysql_url = os.environ.get("MYSQL_URL", "mysql+pymysql://root:@127.0.0.1:3306/investment_data")

    # Set default dates
    today = datetime.datetime.now().strftime("%Y%m%d")
    if not start_date:
        start_date = today
    if not end_date:
        end_date = today

    # Ensure dates are strings (fire may convert them to int)
    start_date = str(start_date)
    end_date = str(end_date)

    # Validate date formats
    try:
        start_dt = datetime.datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.datetime.strptime(end_date, "%Y%m%d")
    except ValueError as e:
        logger.error(f"Invalid date format. Expected YYYYMMDD format: {e}")
        return

    # Ensure start_date <= end_date
    if start_dt > end_dt:
        logger.error(f"start_date ({start_date}) cannot be after end_date ({end_date})")
        return

    # Get trading calendar for the date range
    logger.info("Fetching trading calendar...")
    trade_date_df = get_trade_cal(start_date, end_date)

    if trade_date_df.empty:
        logger.error("Failed to get trading calendar, falling back to all dates")
        logger.debug("Generating calendar dates from start_date to end_date")
        # Fallback: Generate list of all dates
        date_list = []
        current_date = start_dt
        while current_date <= end_dt:
            date_list.append(current_date.strftime("%Y%m%d"))
            current_date += datetime.timedelta(days=1)
        logger.debug(f"Generated {len(date_list)} calendar days")
    else:
        # Sort trading dates from earliest to latest
        trade_date_df = trade_date_df.sort_values("cal_date")
        date_list = trade_date_df["cal_date"].tolist()

        # Filter dates within our range (in case API returned extra dates)
        original_count = len(date_list)
        date_list = [date for date in date_list if start_date <= date <= end_date]
        filtered_count = len(date_list)

        logger.info(f"Found {len(date_list)} trading days in the date range")
        if original_count != filtered_count:
            logger.debug(f"Filtered trading dates: {original_count} -> {filtered_count}")

    logger.info("=== Tushare Brokerage Report Consensus Evaluation ===")
    logger.info(f"Date Range: {start_date} to {end_date}")
    if trade_date_df.empty:
        logger.info(f"Processing: {len(date_list)} calendar days (trading calendar unavailable)")
    else:
        logger.info(f"Processing: {len(date_list)} trading days")
    logger.info(f"MySQL URL: {mysql_url.replace('mysql+pymysql://', 'mysql+pymysql://[HIDDEN]@')}")
    logger.info(f"Force Update: {force_update}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Max Workers: {max_workers}")
    logger.info(f"Dry Run: {dry_run}")

    try:
        engine = create_engine(mysql_url, pool_recycle=3600, pool_pre_ping=True)
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        return

    # Create table if not exists
    try:
        with engine.begin() as conn:
            conn.execute(text(CREATE_TABLE_DDL))
        logger.info("Database table verified/created successfully")
    except Exception as e:
        logger.error(f"Failed to create/verify database table: {e}")
        return

    # Get stocks to process
    try:
        stocks_list = get_stocks_list(engine, stocks)
        if not stocks_list:
            logger.warning("No stocks found to process")
            return
    except Exception as e:
        logger.error(f"Failed to get stocks list: {e}")
        return

    logger.info(f"Processing {len(stocks_list)} stocks...")

    if dry_run:
        logger.info("DRY RUN MODE - No data will be written to database")
        if date_list:
            sample_date = date_list[0]
            fiscal_info = get_fiscal_period_info(sample_date)
            logger.info(f"Sample Date: {sample_date}")
            logger.info(f"Sample Fiscal Info: Year {fiscal_info['current_fiscal_year']}, Period {fiscal_info['current_fiscal_period']}")
            logger.info(f"Total dates to process: {len(date_list)}")
            if len(date_list) <= 10:
                logger.info(f"Dates: {', '.join(date_list)}")
            else:
                logger.info(f"First 5 dates: {', '.join(date_list[:5])}")
                logger.info(f"Last 5 dates: {', '.join(date_list[-5:])}")
        else:
            logger.info("No dates to process")
        return

    all_results = []
    total_processed_count = 0
    total_error_count = 0

    # Process each date in the range
    for current_date in date_list:
        logger.info(f"--- Processing date: {current_date} ---")

        date_results = []
        processed_count = 0
        error_count = 0

        # Process stocks with parallel execution for this date
        logger.debug(f"Starting parallel processing for {current_date} with {max_workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks for this date
            future_to_stock = {
                executor.submit(process_stock_consensus, engine, ts_code, current_date): ts_code
                for ts_code in stocks_list
            }

            logger.debug(f"Submitted {len(future_to_stock)} tasks for {current_date}")

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_stock):
                ts_code = future_to_stock[future]
                try:
                    stock_result = future.result()
                    if stock_result:
                        # Add current date to the result
                        stock_result['eval_date'] = current_date
                        date_results.append(stock_result)
                        logger.debug(f"Successfully processed {ts_code} for {current_date}")
                    else:
                        logger.debug(f"No data generated for {ts_code} on {current_date}")
                    processed_count += 1

                    if processed_count % 50 == 0:
                        logger.info(f"Processed {processed_count}/{len(stocks_list)} stocks for {current_date}")

                except Exception as e:
                    logger.error(f"Error processing {ts_code} for {current_date}: {e}")
                    logger.debug(f"Exception details for {ts_code}: {type(e).__name__}")
                    error_count += 1
                    continue

        # Save results for this date in batches
        if date_results:
            logger.info(f"Collected {len(date_results)} consensus records for {current_date}, saving to database...")

            df = pd.DataFrame(date_results)

            # Data validation and cleaning
            df['last_updated'] = pd.to_datetime(df['last_updated'])
            df = df.replace({np.nan: None})

            # Ensure all required columns exist
            for col in ALL_COLUMNS:
                if col not in df.columns:
                    df[col] = None

            df = df[ALL_COLUMNS]

            written = _upsert_batch(engine, df, chunksize=batch_size)
            logger.info(f"Successfully upserted {written} records for {current_date}")
            all_results.extend(date_results)
        else:
            logger.warning(f"No consensus data generated for {current_date}")

        total_processed_count += processed_count
        total_error_count += error_count

        logger.info(f"Completed {current_date}: {processed_count} stocks processed, {error_count} errors")

    logger.info("=== Evaluation Complete ===")
    logger.info(f"Total records processed: {len(all_results)}")
    logger.info(f"Date range: {start_date} to {end_date}")
    if trade_date_df.empty:
        logger.info(f"Calendar days processed: {len(date_list)}")
    else:
        logger.info(f"Trading days processed: {len(date_list)}")
    logger.info(f"Average stocks per day: {total_processed_count / len(date_list) if date_list else 0:.1f}")
    logger.info(f"Total errors encountered: {total_error_count} stocks")


if __name__ == "__main__":
    # Example usage:
    # python evaluate_brokerage_report.py --start-date 20250101 --end-date 20250105
    # python evaluate_brokerage_report.py --start-date 20250101 --end-date 20250101 --stocks "000001.SZ,000002.SZ"
    # python evaluate_brokerage_report.py --start-date 20250101 --end-date 20250105 --max-workers 8
    # python evaluate_brokerage_report.py --start-date 20250101 --end-date 20250101 --dry-run
    # python evaluate_brokerage_report.py --mysql-url "mysql+pymysql://user:pass@host:port/db"
    fire.Fire(evaluate_brokerage_report)
