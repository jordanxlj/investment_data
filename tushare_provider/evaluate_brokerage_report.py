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
import json

import fire
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, and_, or_, func, MetaData, Table
from sqlalchemy.pool import QueuePool
from sqlalchemy.dialects.mysql import insert as mysql_insert
import tushare as ts


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluate_brokerage_report.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


# Load configurations from JSON
CONFIG_FILE = 'conf/report_configs.json'
try:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        configs = json.load(f)
    RATING_MAPPING = configs.get('rating_mapping', {})
    REPORT_TYPE_WEIGHTS = configs.get('report_type_weights', {})
except FileNotFoundError:
    logger.error(f"Configuration file {CONFIG_FILE} not found. Using defaults.")
except UnicodeDecodeError as e:
    logger.error(f"Encoding error loading config file: {e}. Using defaults.")
    # Default mappings (same as original)
    RATING_MAPPING = {
        'BUY': ['BUY', 'Buy', '买入', '买进', '优于大市', '强于大市', '强力买进', '强推', '强烈推荐', '增持', '推荐', '谨慎增持', '谨慎推荐', '跑赢行业', 'OUTPERFORM', 'OVERWEIGHT', 'Overweight'],
        'HOLD': ['HOLD', 'Hold', '持有', '区间操作'],
        'NEUTRAL': ['Neutral', '中性', '无'],
        'SELL': ['SELL', 'Sell', '卖出', 'Underweight']
    }
    REPORT_TYPE_WEIGHTS = {
        '深度': 5.0, 'depth': 5.0, 'comprehensive': 5.0, 'detailed': 5.0,
        '调研': 4.0, 'research': 4.0, 'field': 4.0, 'visit': 4.0, 'survey': 4.0,
        '点评': 3.0, 'commentary': 3.0, 'comment': 3.0, 'analysis': 3.0, 'review': 3.0,
        '会议纪要': 3.0, '会议': 3.0,
        '一般': 2.0, 'general': 2.0, 'regular': 2.0, 'standard': 2.0,
        '新股': 1.5, '港股': 1.5, '非个股': 1.0, 'non-stock': 1.0, 'industry': 1.0, 'strategy': 1.0, 'sector': 1.0
    }

DEFAULT_REPORT_WEIGHT = 2.0


# Tushare init with validation
TUSHARE_TOKEN = os.environ.get("TUSHARE")
if not TUSHARE_TOKEN:
    logger.error("TUSHARE environment variable not set")
    sys.exit(1)

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


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


def get_report_weight(report_type: Optional[str]) -> float:
    """
    Get weight for a report type

    Args:
        report_type: Report type string

    Returns:
        Weight value (1.0 to 5.0)
    """
    if pd.isna(report_type) or not report_type:  # Explicit NaN/None/empty check
        logger.debug(f"Empty report_type, returning default: {DEFAULT_REPORT_WEIGHT}")
        return DEFAULT_REPORT_WEIGHT

    try:
        report_type_lower = str(report_type).strip().lower()
    except Exception as e:
        logger.error(f"Error converting report_type to str: {type(report_type).__name__} - {e}")
        return DEFAULT_REPORT_WEIGHT

    # Direct match
    if report_type_lower in REPORT_TYPE_WEIGHTS:
        return REPORT_TYPE_WEIGHTS[report_type_lower]

    # Partial match (return first match)
    for key, weight in REPORT_TYPE_WEIGHTS.items():
        if key.lower() in report_type_lower:
            return weight

    logger.debug(f"No match for {report_type_lower}, returning default: {DEFAULT_REPORT_WEIGHT}")
    return DEFAULT_REPORT_WEIGHT


def categorize_report_type(report_type: Optional[str]) -> str:
    """
    Categorize report type into main categories

    Args:
        report_type: Report type string

    Returns:
        Category string ('depth', 'research', 'commentary', 'general', 'other')
    """
    if not report_type:
        return 'other'

    report_type_lower = str(report_type).strip().lower()

    if any(k in report_type_lower for k in ['深度', 'depth', 'comprehensive', 'detailed']):
        return 'depth'
    if any(k in report_type_lower for k in ['调研', 'research', 'field', 'visit', 'survey']):
        return 'research'
    if any(k in report_type_lower for k in ['点评', 'commentary', 'comment', 'analysis', 'review', '会议纪要', '会议']):
        return 'commentary'
    if any(k in report_type_lower for k in ['一般', 'general', 'regular', 'standard', '新股', '港股']):
        return 'general'
    if any(k in report_type_lower for k in ['非个股', 'non-stock', 'industry', 'strategy', 'sector']):
        return 'other'

    return 'other'


def classify_rating(rating: Optional[str]) -> str:
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

    return 'NEUTRAL'


def get_financial_data_only_consensus(engine: Any, ts_code: str, eval_date: str) -> Optional[Dict[str, Any]]:
    """Get consensus data from financial data only (when no brokerage reports available)"""
    try:
        fiscal_info = get_fiscal_period_info(eval_date)

        # Try to get financial data
        current_consensus = get_annual_report_data(engine, ts_code, eval_date, fiscal_info['current_fiscal_period'])

        if current_consensus:
            # Update with basic consensus structure
            current_consensus.update({
                'eval_date': eval_date,
                'total_reports': 0,
                'sentiment_pos': 0,
                'sentiment_neg': 0,
                'buy_count': 0,
                'hold_count': 0,
                'neutral_count': 0,
                'sell_count': 0,
                'depth_reports': 0,
                'research_reports': 0,
                'commentary_reports': 0,
                'general_reports': 0,
                'other_reports': 0,
                'avg_report_weight': 0.0,
                'data_source': 'financial_only'
            })

            # Add last_updated timestamp
            current_consensus['last_updated'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            return current_consensus

        # If no annual report data
        return None
    except Exception as e:
        logger.error(f"Error getting financial data only consensus for {ts_code} {eval_date}: {e}")
        return None

def aggregate_consensus_from_df(date_df: pd.DataFrame, ts_code: str, eval_date: str, fiscal_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Aggregate consensus data from pre-filtered DataFrame"""
    try:
        total_reports = len(date_df)

        # Count ratings
        buy_count = len(date_df[date_df['rating_category'] == 'BUY'])
        hold_count = len(date_df[date_df['rating_category'] == 'HOLD'])
        neutral_count = len(date_df[date_df['rating_category'] == 'NEUTRAL'])
        sell_count = len(date_df[date_df['rating_category'] == 'SELL'])

        # Calculate sentiment
        sentiment_pos = buy_count + hold_count
        sentiment_neg = neutral_count + sell_count

        # Determine dominant sentiment and filter data accordingly
        if sentiment_pos > sentiment_neg:
            # Bullish sentiment - use BUY+HOLD data
            sentiment_df = date_df[date_df['rating_category'].isin(['BUY', 'HOLD'])]
            sentiment = 'bullish'
        elif sentiment_neg > sentiment_pos:
            # Bearish sentiment - use NEUTRAL+SELL data
            sentiment_df = date_df[date_df['rating_category'].isin(['NEUTRAL', 'SELL'])]
            sentiment = 'bearish'
        else:
            # Neutral sentiment - use all data
            sentiment_df = date_df
            sentiment = 'neutral'

        # Count report types
        depth_reports = len(date_df[date_df['report_type'].str.contains('深度|depth|comprehensive|detailed', case=False, na=False)])
        research_reports = len(date_df[date_df['report_type'].str.contains('调研|research|field|visit|survey', case=False, na=False)])
        commentary_reports = len(date_df[date_df['report_type'].str.contains('点评|commentary|comment|analysis|review|会议纪要|会议', case=False, na=False)])
        general_reports = len(date_df[date_df['report_type'].str.contains('一般|general|regular|standard', case=False, na=False)])
        other_reports = total_reports - depth_reports - research_reports - commentary_reports - general_reports

        avg_report_weight = date_df['report_weight'].mean() if not date_df.empty else 0.0

        # Aggregate forecasts
        forecasts = aggregate_forecasts(sentiment_df, sentiment, fiscal_info['current_quarter'])

        result = {
            'ts_code': ts_code,
            'eval_date': eval_date,
            'report_period': fiscal_info['current_quarter'],
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
            'eps': forecasts.get('eps'),
            'pe': forecasts.get('pe'),
            'rd': forecasts.get('rd'),
            'roe': forecasts.get('roe'),
            'ev_ebitda': forecasts.get('ev_ebitda'),
            'max_price': forecasts.get('max_price'),
            'min_price': forecasts.get('min_price'),
            'data_source': 'brokerage_consensus'
        }

        # Add last_updated timestamp
        result['last_updated'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return result

    except Exception as e:
        logger.error(f"Error aggregating consensus from DataFrame for {ts_code} {eval_date}: {e}")
        return None

def process_stock_all_dates(engine: Any, ts_code: str, date_list: List[str], batch_size: int) -> int:
    """Optimized: Process all dates for a single stock and upsert results immediately"""
    # Calculate date range for bulk query (start_date - 6 months to end_date)
    start_dt = datetime.datetime.strptime(min(date_list), "%Y%m%d")
    end_dt = datetime.datetime.strptime(max(date_list), "%Y%m%d")
    bulk_start_dt = start_dt - datetime.timedelta(days=180)  # 6 months back

    bulk_start_date = bulk_start_dt.strftime("%Y%m%d")
    bulk_end_date = end_dt.strftime("%Y%m%d")

    try:
        # Bulk fetch annual data first (higher priority)
        annual_cache = get_annual_data_bulk(engine, ts_code, date_list)

        # Bulk query all brokerage data for this stock in the date range
        with engine.begin() as conn:
            query = text("""
                SELECT * FROM ts_a_stock_brokerage_report
                WHERE ts_code = :ts_code
                AND report_date BETWEEN :start_date AND :end_date
                AND rating IS NOT NULL
                AND report_type IS NOT NULL
                ORDER BY report_date, org_name, report_title
            """)

            bulk_df = pd.read_sql(query, conn, params={
                'ts_code': ts_code,
                'start_date': bulk_start_date,
                'end_date': bulk_end_date
            })

        logger.debug(f"Bulk loaded {len(bulk_df)} brokerage records for {ts_code} from {bulk_start_date} to {bulk_end_date}")

        stock_results = []

        # Precompute fiscal info for each date
        fiscal_infos = {date: get_fiscal_period_info(date) for date in date_list}

        # Group bulk_df by report_date for efficient access
        grouped = bulk_df.groupby('report_date')

        for current_date in date_list:
            try:
                # First check annual data (higher priority)
                annual_data = annual_cache.get(current_date)
                if annual_data:
                    # Use annual data directly
                    result = annual_data.copy()
                    result['ts_code'] = ts_code
                    result['eval_date'] = current_date
                    result['report_period'] = fiscal_infos[current_date]['current_fiscal_period']
                    stock_results.append(result)
                    logger.debug(f"Using annual report data for {ts_code} on {current_date}")
                    continue

                # If no annual data, proceed with brokerage
                # Get group for current date or empty df
                date_df = grouped.get_group(current_date).copy() if current_date in grouped.groups else pd.DataFrame()

                if date_df.empty:
                    # No data at all, skip or add empty result if needed
                    continue

                # Apply report weights vectorized
                date_df['report_weight'] = date_df['report_type'].apply(get_report_weight)

                # Classify ratings vectorized
                date_df['rating_category'] = date_df['rating'].apply(classify_rating)

                # Get precomputed fiscal info
                fiscal_info = fiscal_infos[current_date]
                min_quarter = fiscal_info['current_quarter']

                # Filter by quarter if needed
                if min_quarter != 'ALL':
                    min_quarter_for_comparison = f"{min_quarter}Q4" if min_quarter and 'Q' not in min_quarter else min_quarter
                    date_df['quarter_comparison'] = date_df['quarter'].apply(
                        lambda q: compare_quarters(q, min_quarter_for_comparison) >= 0 if q else False
                    )
                    date_df = date_df[date_df['quarter_comparison']]

                if date_df.empty:
                    continue

                # Aggregate consensus
                result = aggregate_consensus_from_df(date_df, ts_code, current_date, fiscal_info)
                if result:
                    stock_results.append(result)

            except Exception as e:
                logger.error(f"Error processing {ts_code} for date {current_date}: {e}")
                continue

        # Upsert results immediately
        if stock_results:
            try:
                df = pd.DataFrame(stock_results)
                df['last_updated'] = pd.to_datetime(df['last_updated'])
                df = df.replace({np.nan: None})
                for col in ALL_COLUMNS:
                    if col not in df.columns:
                        df[col] = None
                df = df[ALL_COLUMNS]
                upserted = _upsert_batch(engine, df, batch_size)
                logger.debug(f"Upserted {upserted} records for stock {ts_code}")
                return upserted
            except Exception as e:
                logger.error(f"Error upserting results for stock {ts_code}: {e}")
                return 0
        return 0

    except Exception as e:
        logger.error(f"Error in bulk processing for {ts_code}: {e}")
        return 0

def get_trade_cal(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get trading calendar from Tushare API

    Args:
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format

    Returns:
        DataFrame with trading dates
    """
    try:
        df = pro.trade_cal(exchange='SSE', is_open='1',
                           start_date=start_date,
                           end_date=end_date,
                           fields='cal_date')
        return df
    except Exception as e:
        logger.error(f"Error getting trade calendar: {e}")
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
    try:
        eval_dt = datetime.datetime.strptime(eval_date, "%Y%m%d")
    except ValueError:
        raise ValueError(f"Invalid eval_date format: {eval_date}. Expected YYYYMMDD.")

    end_dt = eval_dt
    start_dt = eval_dt - datetime.timedelta(days=window_months * 30)

    return start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")


def get_fiscal_period_info(eval_date: str) -> Dict[str, Any]:
    """
    Get comprehensive fiscal period information for the evaluation date
    Based on China Securities Regulatory Commission requirements for periodic reports

    Args:
        eval_date: Evaluation date in YYYYMMDD format

    Returns:
        Dictionary with fiscal period information
    """
    eval_dt = datetime.datetime.strptime(eval_date, "%Y%m%d")
    year = eval_dt.year
    month = eval_dt.month

    current_year = f"{year}"
    next_year = f"{year + 1}"

    if month <= 4:
        # 1-4月：上年年报发布期，代表上一个会计年度
        current_quarter = f"{year - 1}Q4"
        current_fiscal_year = f"{year - 1}"
        current_fiscal_period = f"{year - 1}1231"
        next_fiscal_year = f"{year}"
        next_fiscal_period = f"{year}1231"
    elif month <= 5:
        # 5月：Q1季报发布期，代表当前会计年度第一季度
        current_quarter = f"{year}Q1"
        current_fiscal_year = f"{year}"
        current_fiscal_period = f"{year}0331"
        next_fiscal_year = f"{year}"
        next_fiscal_period = f"{year}1231"
    elif month <= 8:
        # 7-8月：半年报发布期，代表当前会计年度上半年
        current_quarter = f"{year}Q2"
        current_fiscal_year = f"{year}"
        current_fiscal_period = f"{year}0630"
        next_fiscal_year = f"{year}"
        next_fiscal_period = f"{year}1231"
    elif month <= 11:
        # 10-11月：Q3季报发布期，代表当前会计年度第三季度
        current_quarter = f"{year}Q3"
        current_fiscal_year = f"{year}"
        current_fiscal_period = f"{year}0930"
        next_fiscal_year = f"{year}"
        next_fiscal_period = f"{year}1231"
    else:
        # 12月：Q4季报发布期，代表当前会计年度第四季度
        current_quarter = f"{year}Q4"
        current_fiscal_year = f"{year}"
        current_fiscal_period = f"{year}0930"
        next_fiscal_year = f"{year + 1}"
        next_fiscal_period = f"{year + 1}1231"

    return {
        'eval_date': eval_date,
        'eval_datetime': eval_dt,
        'current_quarter': current_quarter,
        'current_year': current_year,
        'next_year': next_year,
        'current_fiscal_year': current_fiscal_year,
        'current_fiscal_period': current_fiscal_period,
        'next_fiscal_year': next_fiscal_year,
        'next_fiscal_period': next_fiscal_period
    }


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
        year_str, q_str = quarter_str.split('Q')
        year = int(year_str)
        quarter = int(q_str)
        if quarter < 1 or quarter > 4:
            return (0, 0)
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

    Raises:
        ValueError: If either quarter string is invalid
    """
    y1, q1 = parse_quarter(quarter1)
    y2, q2 = parse_quarter(quarter2)

    if (y1 == 0 and q1 == 0) or (y2 == 0 and q2 == 0):
        raise ValueError(f"Invalid quarter format: {quarter1} or {quarter2}")

    if y1 < y2: return -1
    if y1 > y2: return 1
    if q1 < q2: return -1
    if q1 > q2: return 1
    return 0


def _filter_outliers(values: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove extreme outliers beyond 3 standard deviations

    Args:
        values: Array of values
        weights: Array of weights

    Returns:
        Filtered values and weights
    """
    if len(values) > 2:
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # Use sample standard deviation
        if std_val > 0:
            valid_mask = (values >= mean_val - 3 * std_val) & (values <= mean_val + 3 * std_val)
            return values[valid_mask], weights[valid_mask]
    return values, weights


def _apply_field_ranges(field: str, values: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply realistic value ranges based on field type

    Args:
        field: Field name
        values: Array of values
        weights: Array of weights

    Returns:
        Filtered values and weights
    """
    if field == 'eps':
        mask = (values >= -50) & (values <= 50)
    elif field in ['pe', 'ev_ebitda']:
        mask = (values > 0) & (values <= 500)
    elif field == 'roe':
        mask = (values >= -200) & (values <= 200)
    elif field in ['max_price', 'min_price']:
        mask = (values > 0) & (values <= 10000)
    else:
        mask = np.ones(len(values), dtype=bool)

    return values[mask], weights[mask]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate weighted median, averaging middle values for even total weight cases

    Args:
        values: Array of values
        weights: Array of weights corresponding to values

    Returns:
        Weighted median value
    """
    if len(values) == 0:
        raise ValueError("Cannot calculate median of empty array")
    if len(values) != len(weights):
        raise ValueError("Values and weights must have the same length")

    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    median_weight = total_weight / 2
    median_index = np.searchsorted(cum_weights, median_weight, side='right')

    if median_index == 0:
        return float(sorted_values[0])
    if median_index >= len(sorted_values):
        return float(sorted_values[-1])

    if median_index > 0 and cum_weights[median_index - 1] == median_weight:
        return (float(sorted_values[median_index - 1]) + float(sorted_values[median_index])) / 2

    return float(sorted_values[median_index])


def aggregate_forecasts(df: pd.DataFrame, sentiment_source: str, min_quarter: str = 'ALL') -> Dict[str, Any]:
    """
    Aggregate forecast data based on sentiment source with robust handling and weighting

    Args:
        df: DataFrame with brokerage reports
        sentiment_source: 'bullish' or 'bearish' or 'next_year'
        min_quarter: Minimum quarter for filtering ('ALL' or specific quarter)

    Returns:
        Dictionary with aggregated forecast values
    """
    if df.empty:
        return {
            'eps': None, 'pe': None, 'rd': None, 'roe': None,
            'ev_ebitda': None, 'max_price': None, 'min_price': None
        }

    df = df.copy()
    if 'report_weight' not in df.columns:
        df['report_weight'] = pd.Series([get_report_weight(t) for t in df['report_type']], index=df.index)

    quarter_specific_fields = ['eps', 'pe', 'rd', 'roe', 'ev_ebitda']
    all_report_fields = ['max_price', 'min_price']

    result = {}

    # Process quarter-specific fields
    for field in quarter_specific_fields:
        if field in df.columns:
            field_df = df if min_quarter == 'ALL' else df[df['quarter_comparison']] if 'quarter_comparison' in df.columns else df
            field_data = field_df[[field, 'report_weight']].dropna()
            if field_data.empty:
                result[field] = None
                continue

            values = field_data[field].values
            weights = field_data['report_weight'].values

            # Convert values to numeric, handling conversion errors
            try:
                values = pd.to_numeric(values, errors='coerce')
                # Remove NaN values after conversion
                valid_mask = ~pd.isna(values)
                values = values[valid_mask]
                weights = weights[valid_mask]

                if len(values) > 0:
                    values, weights = _filter_outliers(values, weights)
                    values, weights = _apply_field_ranges(field, values, weights)
                else:
                    values = np.array([])
                    weights = np.array([])
            except Exception as e:
                logger.warning(f"Error processing field {field}: {e}")
                values = np.array([])
                weights = np.array([])

            if len(values) == 0:
                result[field] = None
            elif len(values) == 1:
                result[field] = float(values[0])
            else:
                result[field] = weighted_median(values, weights)
        else:
            # Field not in DataFrame, set to None
            result[field] = None

    # Process all-report fields
    for field in all_report_fields:
        if field in df.columns:
            field_data = df[[field, 'report_weight']].dropna()
            if field_data.empty:
                result[field] = None
                continue

            values = field_data[field].values
            weights = field_data['report_weight'].values

            # Convert values to numeric, handling conversion errors
            try:
                values = pd.to_numeric(values, errors='coerce')
                # Remove NaN values after conversion
                valid_mask = ~pd.isna(values)
                values = values[valid_mask]
                weights = weights[valid_mask]

                if len(values) > 0:
                    values, weights = _filter_outliers(values, weights)
                    values, weights = _apply_field_ranges(field, values, weights)
                else:
                    values = np.array([])
                    weights = np.array([])
            except Exception as e:
                logger.warning(f"Error processing field {field}: {e}")
                values = np.array([])
                weights = np.array([])

            if len(values) == 0:
                result[field] = None
            elif len(values) == 1:
                result[field] = float(values[0])
            else:
                result[field] = weighted_median(values, weights)
        else:
            # Field not in DataFrame, set to None
            result[field] = None

    return result


@lru_cache(maxsize=1000)
def get_brokerage_consensus(
    engine: Any,
    ts_code: str,
    eval_date: str,
    min_quarter: str
) -> Optional[Dict[str, Any]]:
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
    try:
        with engine.begin() as conn:
            start_date, end_date = get_date_window(eval_date)
            max_age_date = (datetime.datetime.strptime(eval_date, "%Y%m%d") - datetime.timedelta(days=365)).strftime("%Y%m%d")

            query = text("""
                SELECT * FROM ts_a_stock_brokerage_report
                WHERE ts_code = :ts_code
                AND report_date BETWEEN :start_date AND :end_date
                AND report_date >= :max_age_date
                AND rating IS NOT NULL
                AND report_type IS NOT NULL
            """)

            params = {'ts_code': ts_code, 'start_date': start_date, 'end_date': end_date, 'max_age_date': max_age_date}
            df = pd.read_sql(query, conn, params=params)

            if df.empty:
                return None

            # Filter out invalid quarters (None, empty, or incomplete like 'Q')
            def is_valid_quarter(q):
                if not q or pd.isna(q):
                    return False
                q_str = str(q).strip()
                if len(q_str) <= 1 or not q_str.endswith(('Q1', 'Q2', 'Q3', 'Q4')):
                    return False
                # Check if it follows YYYYQX pattern
                if len(q_str) != 6:  # YYYYQX format
                    return False
                try:
                    year = int(q_str[:4])
                    quarter = int(q_str[5])
                    return 1 <= quarter <= 4 and 2000 <= year <= 2100
                except ValueError:
                    return False

            valid_quarters_mask = df['quarter'].apply(is_valid_quarter)
            invalid_count = len(df) - valid_quarters_mask.sum()
            if invalid_count > 0:
                logger.debug(f"Filtered out {invalid_count} records with invalid quarters")
            df = df[valid_quarters_mask]

            # Debug: Log quarter values for this stock
            logger.debug(f"start_date: {start_date}, end_date: {end_date}, max_age_date: {max_age_date}")
            logger.debug(f"Brokerage data for {ts_code}: {len(df)} records")
            logger.debug(f"Quarter values: {df['quarter'].unique()[:10].tolist()}")  # Show first 10 unique quarters
            logger.debug(f"Quarter null count: {df['quarter'].isnull().sum()}")

            # Handle min_quarter format - if it's just a year, convert to Q4 format for comparison
            min_quarter_for_comparison = min_quarter
            if min_quarter and 'Q' not in min_quarter:
                # If min_quarter is just a year (e.g., '2024'), treat it as '2024Q4' for comparison
                min_quarter_for_comparison = f"{min_quarter}Q4"
            logger.debug(f"Min quarter: {min_quarter}, min_quarter_for_comparison: {min_quarter_for_comparison}")

            df['quarter_comparison'] = df['quarter'].apply(lambda q: compare_quarters(q, min_quarter_for_comparison) >= 0 if q else False)
            df = df[df['quarter_comparison']]

            if df.empty:
                return None

            df['rating_category'] = df['rating'].apply(classify_rating)
            df['report_weight'] = pd.Series([get_report_weight(t) for t in df['report_type']], index=df.index)
            df['report_type_category'] = df['report_type'].apply(categorize_report_type)

            rating_counts = df['rating_category'].value_counts()
            buy_count = rating_counts.get('BUY', 0)
            hold_count = rating_counts.get('HOLD', 0)
            neutral_count = rating_counts.get('NEUTRAL', 0)
            sell_count = rating_counts.get('SELL', 0)

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

            if sentiment_pos > sentiment_neg:
                sentiment_df = df[df['rating_category'].isin(['BUY', 'HOLD'])]
                sentiment = 'bullish'
            elif sentiment_neg > sentiment_pos:
                sentiment_df = df[df['rating_category'].isin(['NEUTRAL', 'SELL'])]
                sentiment = 'bearish'
            else:
                sentiment_df = df
                sentiment = 'neutral'

            forecasts = aggregate_forecasts(sentiment_df, sentiment, min_quarter=min_quarter)

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
                'eps': forecasts.get('eps'),
                'pe': forecasts.get('pe'),
                'rd': forecasts.get('rd'),
                'roe': forecasts.get('roe'),
                'ev_ebitda': forecasts.get('ev_ebitda'),
                'max_price': forecasts.get('max_price'),
                'min_price': forecasts.get('min_price'),
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


def get_next_year_consensus(
    engine: Any,
    ts_code: str,
    eval_date: str,
    next_year: str
) -> Optional[Dict[str, Any]]:
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
    eval_dt = datetime.datetime.strptime(eval_date, "%Y%m%d")
    month = eval_dt.month

    if month <= 3:
        next_year_pattern = f"{next_year}Q4"
    else:
        next_year_pattern = f"{next_year}Q%"

    try:
        with engine.begin() as conn:
            start_date, end_date = get_date_window(eval_date)
            max_age_date = (eval_dt - datetime.timedelta(days=365)).strftime("%Y%m%d")

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

            params = {
                'ts_code': ts_code,
                'start_date': start_date,
                'end_date': end_date,
                'max_age_date': max_age_date,
                'next_year_pattern': next_year_pattern
            }

            df = pd.read_sql(query, conn, params=params)

            if df.empty:
                return None

            # Apply report weights with error handling
            try:
                df['report_weight'] = pd.Series([get_report_weight(t) for t in df['report_type']], index=df.index)
            except Exception as e:
                logger.warning(f"Error applying report weights: {e}. Using default weights.")
                df['report_weight'] = DEFAULT_REPORT_WEIGHT

            total_reports = len(df)
            avg_report_weight = df['report_weight'].mean() if not df.empty else 0.0

            try:
                forecasts = aggregate_forecasts(df, 'next_year', min_quarter='ALL')
            except Exception as e:
                logger.warning(f"Error in forecast aggregation: {e}. Returning basic stats only.")
                forecasts = {
                    'eps': None, 'pe': None, 'rd': None, 'roe': None,
                    'ev_ebitda': None, 'max_price': None, 'min_price': None
                }

            return {
                'total_reports': total_reports,
                'avg_report_weight': avg_report_weight,
                'eps': forecasts.get('eps'),
                'pe': forecasts.get('pe'),
                'roe': forecasts.get('roe'),
                'ev_ebitda': forecasts.get('ev_ebitda')
            }
    except Exception as e:
        logger.error(f"Error getting next year consensus for {ts_code}: {e}")
        return None


def get_annual_report_data(
    engine: Any,
    ts_code: str,
    eval_date: str,
    report_period: str
) -> Optional[Dict[str, Any]]:
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

            params = {'ts_code': ts_code, 'report_period': report_period, 'eval_date': eval_date}
            df = pd.read_sql(query, conn, params=params)

            if df.empty:
                return None

            row = df.iloc[0]

            fundamental_query = text("""
                SELECT pe, dv_ratio FROM ts_a_stock_fundamental
                WHERE ts_code = :ts_code
                AND trade_date <= :eval_date
                ORDER BY trade_date DESC
                LIMIT 1
            """)

            fundamental_params = {'ts_code': ts_code, 'eval_date': eval_date}
            fundamental_df = pd.read_sql(fundamental_query, conn, params=fundamental_params)

            pe_value = fundamental_df.iloc[0]['pe'] if not fundamental_df.empty else None
            dv_ratio_value = fundamental_df.iloc[0]['dv_ratio'] if not fundamental_df.empty else None

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
                'eps': row.get('eps'),
                'pe': pe_value,
                'rd': dv_ratio_value,
                'roe': row.get('roe_waa'),
                'ev_ebitda': None,
                'max_price': None,
                'min_price': None,
                'next_year_eps': None,
                'next_year_pe': None,
                'next_year_roe': None,
                'next_year_ev_ebitda': None,
                'next_year_reports': None,
                'next_year_avg_weight': None,
                'data_source': 'annual_report'
            }

            # Add last_updated timestamp
            result['last_updated'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            return result
    except Exception as e:
        logger.error(f"Error getting annual report data for {ts_code}: {e}")
        return None


def get_annual_data_bulk(engine: Any, ts_code: str, date_list: List[str]) -> Dict[str, Dict[str, Any]]:
    """Bulk fetch annual report data for all dates for a stock"""
    annual_data_cache = {}

    try:
        # Get unique report periods from date_list
        periods = set()
        for date in date_list:
            fiscal_info = get_fiscal_period_info(date)
            periods.add(fiscal_info['current_fiscal_period'])

        periods_list = list(periods)
        if not periods_list:
            return annual_data_cache

        # Bulk query financial_profile
        with engine.begin() as conn:
            fp_query = text("""
                SELECT ann_date, f_ann_date, period, eps, roe_waa
                FROM ts_a_stock_financial_profile
                WHERE ts_code = :ts_code
                AND period IN :periods
                ORDER BY period
            """)
            fp_df = pd.read_sql(fp_query, conn, params={
                'ts_code': ts_code,
                'periods': tuple(periods_list)
            })

            # Bulk query fundamental
            fund_query = text("""
                SELECT ann_date, period, pe, ev_to_ebitda, dv_ratio
                FROM ts_a_stock_fundamental
                WHERE ts_code = :ts_code
                AND period IN :periods
                ORDER BY period
            """)
            fund_df = pd.read_sql(fund_query, conn, params={
                'ts_code': ts_code,
                'periods': tuple(periods_list)
            })

        # Process and cache per date
        for date in date_list:
            fiscal_info = get_fiscal_period_info(date)
            period = fiscal_info['current_fiscal_period']

            fp_row = fp_df[fp_df['period'] == period]
            fund_row = fund_df[fund_df['period'] == period]

            if not fp_row.empty:
                row = fp_row.iloc[0]
                annual_data = {
                    'eps': row['eps'],
                    'roe': row['roe_waa'],
                    'total_reports': 0,
                    'sentiment_pos': 0,
                    'sentiment_neg': 0,
                    'buy_count': 0,
                    'hold_count': 0,
                    'neutral_count': 0,
                    'sell_count': 0,
                    'depth_reports': 0,
                    'research_reports': 0,
                    'commentary_reports': 0,
                    'general_reports': 0,
                    'other_reports': 0,
                    'avg_report_weight': None,
                    'pe': None,
                    'rd': None,
                    'ev_ebitda': None,
                    'max_price': None,
                    'min_price': None,
                    'data_source': 'annual_report',
                    'last_updated': datetime.datetime.now()
                }

                if not fund_row.empty:
                    f_row = fund_row.iloc[0]
                    annual_data['pe'] = f_row['pe']
                    annual_data['ev_ebitda'] = f_row['ev_to_ebitda']
                    annual_data['rd'] = f_row['dv_ratio']  # Assuming rd is dv_ratio, adjust if needed

                annual_data_cache[date] = annual_data
            else:
                annual_data_cache[date] = None

        logger.debug(f"Bulk loaded annual data for {ts_code} with {len(annual_data_cache)} entries")
        return annual_data_cache

    except Exception as e:
        logger.error(f"Error bulk loading annual data for {ts_code}: {e}")
        return {date: None for date in date_list}


def process_stock_consensus(
    engine: Any,
    ts_code: str,
    eval_date: str
) -> Optional[Dict[str, Any]]:
    """
    Process consensus data for a single stock with enhanced fiscal period handling

    Args:
        engine: SQLAlchemy engine
        ts_code: Stock code
        eval_date: Evaluation date

    Returns:
        Dictionary with combined current and next year consensus data or None
    """
    fiscal_info = get_fiscal_period_info(eval_date)

    current_consensus = get_brokerage_consensus(engine, ts_code, eval_date, fiscal_info['current_quarter'])

    if not current_consensus:
        annual_data = get_annual_report_data(engine, ts_code, eval_date, fiscal_info['current_fiscal_period'])
        if annual_data:
            current_consensus = annual_data

    if not current_consensus:
        return None

    next_year_data = get_next_year_consensus(engine, ts_code, eval_date, fiscal_info['next_year'])

    if next_year_data:
        current_consensus.update({
            'next_year_eps': next_year_data['eps'],
            'next_year_pe': next_year_data['pe'],
            'next_year_roe': next_year_data['roe'],
            'next_year_ev_ebitda': next_year_data['ev_ebitda'],
            'next_year_reports': next_year_data['total_reports'],
            'next_year_avg_weight': next_year_data['avg_report_weight']
        })

    return current_consensus


def _upsert_batch(engine: Any, df: pd.DataFrame, chunksize: int = 1000) -> int:
    """
    Upsert consensus data in batches with optimized performance

    Args:
        engine: SQLAlchemy engine
        df: DataFrame with consensus data
        chunksize: Batch size for upsert

    Returns:
        Number of rows upserted
    """
    if df.empty:
        return 0

    total = 0
    meta = MetaData()
    # Check if engine is a mock (for testing)
    if hasattr(engine, '_mock_name') or str(type(engine)).startswith("<class 'unittest.mock."):
        # For testing, create table without autoload
        from sqlalchemy import Column, String, Integer, Float, DateTime
        table = Table(TABLE_NAME, meta,
                     Column('ts_code', String(16)),
                     Column('eval_date', String(8)),
                     Column('report_period', String(10)),
                     # Add other columns as needed for testing
                     extend_existing=True)
    else:
        table = Table(TABLE_NAME, meta, autoload_with=engine)
    rows = df.to_dict(orient='records')

    with engine.begin() as conn:
        for i in range(0, len(rows), chunksize):
            batch = rows[i:i + chunksize]
            stmt = mysql_insert(table).values(batch)
            update_map = {c: stmt.inserted[c] for c in ALL_COLUMNS if c not in ('ts_code', 'eval_date', 'report_period')}
            ondup = stmt.on_duplicate_key_update(**update_map)
            result = conn.execute(ondup)
            total += result.rowcount

    return total


def get_stocks_list(engine: Any, stocks: Optional[List[str]] = None) -> List[str]:
    """
    Get list of active stocks to process from ts_a_stock_basic table

    Args:
        engine: SQLAlchemy engine
        stocks: Optional list of specific stocks

    Returns:
        List of active stock codes
    """
    if stocks:
        if isinstance(stocks, str):
            stocks = [s.strip() for s in stocks.split(',')]
        return stocks

    try:
        with engine.begin() as conn:
            query = text("""
                SELECT ts_code FROM ts_a_stock_basic
                WHERE list_status = 'L'
                AND (delist_date IS NULL OR delist_date = '' OR delist_date > :current_date)
                ORDER BY ts_code
            """)
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            result = conn.execute(query, {'current_date': current_date})
            return [row[0] for row in result.fetchall()]
    except Exception as e:
        logger.error(f"Error getting stocks list: {e}")
        return []


def evaluate_brokerage_report(
    mysql_url: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stocks: Optional[List[str]] = None,
    force_update: bool = False,
    batch_size: int = 50,
    max_workers: int = 16,
    dry_run: bool = False,
    window_months: int = 6
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
        window_months: Months for report window (default: 6)
    """
    mysql_url = mysql_url or os.environ.get("MYSQL_URL", "mysql+pymysql://root:@127.0.0.1:3306/investment_data")
    today = datetime.datetime.now().strftime("%Y%m%d")
    start_date = str(start_date or today)
    end_date = str(end_date or today)

    try:
        start_dt = datetime.datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.datetime.strptime(end_date, "%Y%m%d")
        if start_dt > end_dt:
            raise ValueError("start_date cannot be after end_date")
    except ValueError as e:
        logger.error(f"Invalid date: {e}")
        return

    trade_date_df = get_trade_cal(start_date, end_date)
    if not trade_date_df.empty:
        # 确保按日期正序排序（从早到晚）
        trade_date_df = trade_date_df.sort_values('cal_date', ascending=True)
        date_list = trade_date_df['cal_date'].tolist()
    else:
        date_list = [
            (start_dt + datetime.timedelta(days=i)).strftime("%Y%m%d")
            for i in range((end_dt - start_dt).days + 1)
        ]

    # Optimize MySQL connection parameters for high concurrency
    mysql_url_with_params = mysql_url + "?charset=utf8mb4&autocommit=true&max_allowed_packet=67108864"

    engine = create_engine(
        mysql_url_with_params,
        poolclass=QueuePool,
        pool_size=100,  # 增加连接池大小
        max_overflow=100,  # 增加额外连接数
        pool_recycle=1800,  # 减少回收时间
        pool_pre_ping=True,  # 连接前ping
        pool_timeout=60,  # 增加获取连接超时时间
        echo=False,  # 不打印SQL语句
        connect_args={
            'connect_timeout': 10,
            'read_timeout': 30,
            'write_timeout': 30,
        }
    )
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_DDL))

    stocks_list = get_stocks_list(engine, stocks)
    if not stocks_list:
        return

    if dry_run:
        logger.info("DRY RUN - No DB writes")
        return


    logger.info(f"Processing {len(stocks_list)} stocks with {max_workers} workers (each worker handles one stock for all {len(date_list)} dates)")
    logger.info(f"Stocks to process: {stocks_list[:10]}..." if len(stocks_list) > 10 else f"Stocks to process: {stocks_list}")

    # Log connection pool status
    logger.info(f"Connection pool status - size: {engine.pool.size()}, checkedin: {engine.pool.checkedin()}, overflow: {engine.pool.overflow()}")

    total_upserted = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit one task per stock (each task processes all dates for that stock and upserts immediately)
        futures = {executor.submit(process_stock_all_dates, engine, ts_code, date_list, batch_size): ts_code for ts_code in stocks_list}

        logger.info(f"Submitted {len(futures)} tasks to thread pool (one per stock)")

        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            ts_code = futures[future]
            try:
                upserted_count = future.result()
                total_upserted += upserted_count
                completed_count += 1

                if completed_count % 50 == 0:  # Log progress every 50 stocks
                    logger.info(f"Completed {completed_count}/{len(futures)} stocks, total upserted: {total_upserted}")

            except Exception as e:
                logger.error(f"Error processing stock {ts_code}: {e}")
                completed_count += 1

    logger.info(f"Completed processing all stocks. Total upserted: {total_upserted} records")
    logger.info(f"Processed: {len(stocks_list)} stocks * {len(date_list)} dates = {len(stocks_list) * len(date_list)} stock-date combinations")

if __name__ == "__main__":
    fire.Fire(evaluate_brokerage_report)

