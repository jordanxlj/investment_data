"""
TTM (Trailing Twelve Months) and CAGR Calculation Module

This module calculates TTM financial metrics and CAGR from ts_a_stock_financial_profile
and updates final_a_stock_comb_info table.

USAGE:
    # Run TTM/CAGR calculation
    python update_annual_report_ttm.py --start_date 2024-01-01 --end_date 2024-12-31

    # Create essential indexes for ts_a_stock_financial_profile
    python update_annual_report_ttm.py create_indexes

    # Test date format conversion
    python update_annual_report_ttm.py test_dates

CRITICAL INDEXES REQUIRED FOR ts_a_stock_financial_profile:
===============================================================================

必需的索引（按优先级排序）：

1. PRIMARY KEY (ts_code, ann_date, report_period) - 如果不存在则创建
   - 原因：唯一标识每条财务记录

2. INDEX idx_ts_code_ann_date (ts_code, ann_date) - 最重要！
   - 原因：覆盖主要的查询模式 (ts_code + ann_date范围)
   - 影响：提升 90% 的查询性能

3. INDEX idx_ts_code_report_period_ann_date (ts_code, report_period, ann_date)
   - 原因：完全覆盖所有WHERE条件和ORDER BY
   - 影响：查询性能最优化

4. INDEX idx_ann_date (ann_date)
   - 原因：纯日期范围查询
   - 影响：辅助日期过滤

5. INDEX idx_report_period (report_period)
   - 原因：报告期模式匹配 (LIKE查询)
   - 影响：提升报告期过滤性能

===============================================================================

Key Features:
- Dynamic weighting based on report type and availability
- TTM calculations for 10+ financial metrics
- 3-year CAGR calculations for revenue and net income
- Robust error handling and data validation

Weighting Logic:
- Q1: Annual report available -> use annual only
- Q1: Annual report not available -> mix of previous periods (0.25 each)
- Q2: Q2 report available -> annual 0.5 + Q1 0.25 + Q2 0.25
- Q2: Q2 report not available -> annual 0.75 + Q1 0.25
- Q3/Q4: equal weight distribution (0.25 each)
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from sqlalchemy import create_engine, text, QueuePool
import logging
from typing import Dict, List, Optional, Tuple, Union
import json
import fire
import concurrent.futures

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('update_annual_report_ttm.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class TTMCalculator:
    """TTM and CAGR Calculator for financial data"""

    def __init__(self, mysql_url: str = "mysql+pymysql://root:@127.0.0.1:3306/investment_data", annual_config_path: str = 'conf/annual_config.json'):
        """Initialize calculator with configuration"""
        self.annual_config = self.load_annual_config(annual_config_path)
        self.engine = self.create_db_engine(mysql_url)
        self.report_periods = {
            'annual': '%-12-31',
            'q1': '%-03-31',
            'q2': '%-06-30',
            'q3': '%-09-30'
        }

    def load_annual_config(self, config_path: str) -> Dict:
        """Load annual report CAGR configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded CAGR configuration for {len(config.get('cagr_metrics', {}))} metrics")
            return config
        except Exception as e:
            logger.error(f"Failed to load annual config: {e}")
            raise

    def create_db_engine(self, mysql_url: str):
        """Create SQLAlchemy database engine with optimized connection pooling"""
        # Optimize MySQL connection parameters for high concurrency
        mysql_url_with_params = mysql_url + "?charset=utf8mb4&autocommit=true&max_allowed_packet=67108864"

        return create_engine(
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

    def process_updates_by_stock(
        self,
        start_date: str = None,
        end_date: str = None,
        stocks: List[str] = None,
        batch_size: int = 50,
        max_workers: int = 8
    ) -> None:
        """Process updates by stock (similar to evaluate_brokerage_report)"""
        try:
            # Set default dates (YYYY-MM-DD format for consistency with DATE type columns)
            if start_date is None or end_date is None:
                today = datetime.now().strftime('%Y-%m-%d')
                start_date = start_date or today
                end_date = end_date or today

            # Validate dates
            try:
                start_dt = pd.to_datetime(start_date, format='%Y-%m-%d')
                end_dt = pd.to_datetime(end_date, format='%Y-%m-%d')
                if start_dt > end_dt:
                    raise ValueError("start_date cannot be after end_date")
            except ValueError as e:
                logger.error(f"Invalid date: {e}")
                return

            # Get date list
            date_list = self.get_date_list(start_date, end_date)
            if not date_list:
                logger.info("No dates to process")
                return

            # Get current quarter info and weights
            current_quarter, current_month, current_year = self.get_current_quarter_info()
            logger.info(f"Current quarter: {current_quarter}")

            # Get stocks list
            stocks_list = self.get_stocks_list(stocks)
            if not stocks_list:
                logger.info("No stocks to process")
                return

            logger.info(f"Processing {len(stocks_list)} stocks with {max_workers} workers")
            logger.info(f"Each worker queries its own stock data and processes all dates")

            # Process stocks in parallel (similar to evaluate_brokerage_report)
            # Each worker will query its own stock data to avoid memory issues
            all_updates = []
            total_processed = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit one task per stock - each worker queries its own data
                futures = {
                    executor.submit(self.process_single_stock_from_batch, ts_code, start_date, end_date): ts_code
                    for ts_code in stocks_list
                }

                # Process completed tasks
                for future in concurrent.futures.as_completed(futures):
                    ts_code = futures[future]
                    try:
                        updates = future.result()
                        all_updates.extend(updates)
                        total_processed += 1

                        if total_processed % 100 == 0:
                            logger.info(f"Completed {total_processed}/{len(stocks_list)} stocks")

                    except Exception as e:
                        logger.error(f"Failed to process {ts_code}: {e}")

            # Update database in batches
            logger.info(f"Generated {len(all_updates)} total updates")

            if all_updates:
                batch_size_db = 1000
                for i in range(0, len(all_updates), batch_size_db):
                    batch_updates = all_updates[i:i + batch_size_db]
                    self.update_final_table(batch_updates)
                    logger.info(f"Updated batch {i//batch_size_db + 1}: {len(batch_updates)} records")

            logger.info("TTM and CAGR update completed successfully")

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

    def get_date_list(self, start_date: str, end_date: str) -> List[str]:
        """Get list of dates to process"""
        try:
            # Get trading calendar
            query = text("""
            SELECT date
            FROM ts_trade_day_calendar
            WHERE date >= DATE(:start_date)
              AND date <= DATE(:end_date)
              AND is_open = 1
            ORDER BY date
            """)
            dates_df = pd.read_sql(query, self.engine, params={"start_date": start_date, "end_date": end_date})
            date_list = dates_df['date'].tolist()

            if not date_list:
                # Fallback: generate date range if no trading calendar
                start_dt = pd.to_datetime(start_date, format='%Y-%m-%d')
                end_dt = pd.to_datetime(end_date, format='%Y-%m-%d')
                date_list = [
                    (start_dt + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
                    for i in range((end_dt - start_dt).days + 1)
                ]

            logger.info(f"Processing {len(date_list)} trading dates from {start_date} to {end_date}")
            return date_list

        except Exception as e:
            logger.error(f"Failed to get date list: {e}")
            return []

    def get_stocks_list(self, stocks: List[str] = None) -> List[str]:
        """Get list of stocks to process"""
        try:
            if stocks:
                # Handle stocks parameter - it could be a list or a single string
                if isinstance(stocks, str):
                    # If it's a single string, split by comma
                    stocks_list = [s.strip().upper() for s in stocks.split(',') if s.strip()]
                elif isinstance(stocks, list):
                    # If it's already a list, process each item
                    stocks_list = []
                    for stock in stocks:
                        if isinstance(stock, str):
                            # Split each string by comma in case of nested strings
                            stocks_list.extend([s.strip().upper() for s in stock.split(',') if s.strip()])
                        else:
                            stocks_list.append(str(stock).upper())
                else:
                    stocks_list = []

                logger.debug("Using provided stocks list: %s", stocks_list)
                logger.info(f"Using provided stocks list: {len(stocks_list)} stocks")
            else:
                # Get all stocks from ts_link_table
                query = text("""
                SELECT DISTINCT w_symbol as symbol
                FROM ts_link_table
                WHERE w_symbol IS NOT NULL
                ORDER BY w_symbol
                """)
                stocks_df = pd.read_sql(query, self.engine)
                stocks_list = stocks_df['symbol'].tolist()
                logger.info(f"Retrieved {len(stocks_list)} stocks from database")

            return stocks_list
        except Exception as e:
            logger.error(f"Failed to get stocks list: {e}")
            return []

    def get_current_quarter_info(self) -> Tuple[int, int, int]:
        """Get current quarter, month, and year information"""
        today = datetime.now()
        quarter = (today.month - 1) // 3 + 1
        month = today.month
        year = today.year
        return quarter, month, year

    def process_single_stock_from_batch(self, ts_code: str, start_date: str, end_date: str) -> List[Dict]:
        """Process a single stock for date range by querying its data individually with smart calculation skipping"""
        try:
            updates = []

            # Query financial data for this specific stock in the date range
            stock_financial_df = self.get_financial_data_for_single_stock(ts_code, start_date, end_date)

            if stock_financial_df.empty:
                logger.warning(f"No financial data found for {ts_code} in date range {start_date}-{end_date}")
                return updates

            # Get all unique announcement dates for this stock (sorted)
            announcement_dates = sorted(stock_financial_df['ann_date'].unique())
            logger.debug(f"{ts_code} has {len(announcement_dates)} financial reports in date range")

            # Get the actual trading dates for this stock within the date range
            stock_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            date_list = [d.strftime('%Y-%m-%d') for d in stock_dates]

            # Smart calculation: skip dates where no new financial data was released
            last_calculation_date = None
            last_ttm_metrics = None
            last_cagr_metrics = None

            for target_date in date_list:
                try:
                    target_date_dt = pd.to_datetime(target_date, format='%Y-%m-%d')

                    # Find the most recent announcement date before or on target_date
                    recent_announcement_dates = [d for d in announcement_dates if d < target_date_dt]

                    if not recent_announcement_dates:
                        # No financial data available yet for this target_date
                        logger.debug(f"Skipping {ts_code} on {target_date}: no financial data available")
                        continue

                    most_recent_announcement = max(recent_announcement_dates)

                    # Check if we have new financial data since last calculation
                    has_new_data = (last_calculation_date is None or
                                  most_recent_announcement > last_calculation_date)

                    if not has_new_data and last_ttm_metrics is not None and last_cagr_metrics is not None:
                        # No new financial data, use previous calculation results
                        logger.debug(f"Reusing previous calculation for {ts_code} on {target_date}")
                        update_data = {
                            'tradedate': target_date,
                            'symbol': ts_code,
                            **last_ttm_metrics,
                            **last_cagr_metrics
                        }
                        updates.append(update_data)
                        continue

                    # New financial data available, perform calculation
                    logger.debug(f"Calculating new metrics for {ts_code} on {target_date} (new data: {most_recent_announcement.strftime('%Y-%m-%d')})")

                    # Filter data for this specific date - annual data for CAGR
                    annual_df = self.get_annual_data_for_cagr(stock_financial_df, [ts_code], target_date)

                    # Filter data for this specific date - quarterly data for TTM
                    quarterly_df = self.get_quarterly_data_for_ttm(stock_financial_df, [ts_code], target_date)

                    # Calculate TTM metrics using quarterly data
                    ttm_metrics = self.calculate_ttm_metrics(quarterly_df, target_date)

                    # Calculate CAGR using annual data
                    cagr_metrics = self.calculate_cagr(annual_df)

                    # Store results for potential reuse
                    last_calculation_date = most_recent_announcement
                    last_ttm_metrics = ttm_metrics.copy()
                    last_cagr_metrics = cagr_metrics.copy()

                    # Combine all metrics
                    update_data = {
                        'tradedate': target_date,
                        'symbol': ts_code,
                        **ttm_metrics,
                        **cagr_metrics
                    }

                    updates.append(update_data)

                except Exception as e:
                    logger.error(f"Failed to process {ts_code} for date {target_date}: {e}")
                    continue

            logger.info(f"Processed {ts_code}: {len(updates)} updates generated (optimized calculation)")
            return updates

        except Exception as e:
            logger.error(f"Failed to process stock {ts_code}: {e}")
            return []

    def get_financial_data_for_single_stock(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get all financial data for a single stock within date range"""
        try:
            # Calculate date range for TTM (12 months) and CAGR (max periods + 1 years)
            start_dt = pd.to_datetime(start_date, format='%Y-%m-%d')
            end_dt = pd.to_datetime(end_date, format='%Y-%m-%d')

            # For CAGR: find maximum periods needed
            cagr_metrics = self.annual_config.get('cagr_metrics', {})
            max_periods = 0
            for config in cagr_metrics.values():
                periods = config.get('periods', 3)
                max_periods = max(max_periods, periods)

            # Calculate start dates for both TTM and CAGR
            start_date_ttm = start_dt - pd.DateOffset(months=12)
            start_date_cagr = start_dt - pd.DateOffset(years=max_periods + 1)

            # Use the earliest start date (DATE type format)
            query_start_date = min(start_date_ttm, start_date_cagr)
            query_start_str = query_start_date.strftime('%Y-%m-%d')
            query_end_str = end_dt.strftime('%Y-%m-%d')

            # Get all required fields for both CAGR and TTM calculations
            required_fields = set()

            # Add CAGR fields (support both single field and multiple fields)
            for metric_config in cagr_metrics.values():
                source_field = metric_config.get('source_field')
                source_fields = metric_config.get('source_fields')

                if source_field:
                    required_fields.add(source_field)
                elif source_fields:
                    # source_fields can be a list or set
                    if isinstance(source_fields, (list, set)):
                        required_fields.update(source_fields)
                    elif isinstance(source_fields, dict):
                        # If it's a dict, add all values
                        required_fields.update(source_fields.values())
                    else:
                        logger.warning("Invalid source_fields format for %s: %s", str(metric_config), str(source_fields))

            # Add TTM fields
            ttm_metrics_config = self.annual_config.get('ttm_metrics', {})
            for metric_config in ttm_metrics_config.values():
                source_field = metric_config.get('source_field')
                if source_field:
                    required_fields.add(source_field)

            # Build SELECT clause dynamically
            select_fields = [
                "financial.ts_code",
                "financial.report_period",
                "financial.ann_date"
            ]

            # Add required financial fields
            for field in sorted(required_fields):
                select_fields.append(f"financial.{field}")

            select_clause = ",\n            ".join(select_fields)

            query = text(f"""
            SELECT
                {select_clause}
            FROM ts_a_stock_financial_profile financial
            WHERE financial.ts_code = :ts_code
                AND financial.ann_date >= DATE(:query_start_date)
                AND financial.ann_date <= DATE(:query_end_date)
                AND (
                    DATE_FORMAT(financial.report_period, '%m-%d') IN ('12-31', '03-31', '06-30', '09-30')
                )
            ORDER BY financial.ann_date DESC
            """)

            df = pd.read_sql(query, self.engine, params={
                "ts_code": ts_code,
                "query_start_date": query_start_str,
                "query_end_date": query_end_str
            })
            if not df.empty:
                df['ann_date'] = pd.to_datetime(df['ann_date'])
                df['report_year'] = df['report_period'].str[:4].astype(int)
                df['report_quarter'] = df['report_period'].str[-5:-3]

            logger.debug("Retrieved %d financial records for %s", len(df), ts_code)
            return df

        except Exception as e:
            logger.error("Failed to get financial data for %s: %s", ts_code, str(e))
            raise

    def get_target_dates(self) -> pd.DataFrame:
        """Get dates that need TTM/CAGR updates"""
        query = text("""
        SELECT DISTINCT trade_date
        FROM ts_a_stock_fundamental ts_raw
        LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
        WHERE STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') > COALESCE(
            (SELECT MAX(tradedate) FROM final_a_stock_comb_info), '2008-01-01'
        )
        ORDER BY trade_date
        """)

        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"Found {len(df)} target dates for update")
            return df
        except Exception as e:
            logger.error(f"Failed to get target dates: {e}")
            raise

    def get_annual_data_for_cagr(self, stock_df: pd.DataFrame, ts_codes: List[str], target_date: str) -> pd.DataFrame:
        """Filter annual data from stock data for CAGR calculation"""
        if stock_df.empty:
            return pd.DataFrame()

        # Calculate the target year from target_date
        target_date_dt = pd.to_datetime(target_date, format='%Y-%m-%d')
        target_year = target_date_dt.year

        # Get CAGR configuration to determine required years
        cagr_metrics = self.annual_config.get('cagr_metrics', {})

        # Find the maximum periods needed across all metrics
        max_periods = 0
        for config in cagr_metrics.values():
            periods = config.get('periods', 3)
            max_periods = max(max_periods, periods)

        # Get enough years of data: max_periods + 1 (for start and end values)
        years_to_include = [str(target_year - i) for i in range(max_periods + 1)]

        # Filter for required stocks, date range, and annual reports
        filtered_df = stock_df[
            (stock_df['ts_code'].isin(ts_codes)) &
            (stock_df['ann_date'] <= target_date_dt) &
            (stock_df['report_period'].isin([f"{year}-12-31" for year in years_to_include]))
        ].copy()

        logger.debug("Filtered %d annual records for CAGR calculation (years: %s)", len(filtered_df), ', '.join(years_to_include))
        return filtered_df

    def get_quarterly_data_for_ttm(self, stock_df: pd.DataFrame, ts_codes: List[str], target_date: str) -> pd.DataFrame:
        """Filter quarterly and annual data from stock data for TTM calculation"""
        if stock_df.empty:
            return pd.DataFrame()

        # Convert target_date to proper format and calculate date range
        target_date_dt = pd.to_datetime(target_date, format='%Y-%m-%d')
        start_date = target_date_dt - pd.DateOffset(months=12)

        # Filter for required stocks and date range
        filtered_df = stock_df[
            (stock_df['ts_code'].isin(ts_codes)) &
            (stock_df['ann_date'] <= target_date_dt) &
            (stock_df['ann_date'] >= start_date) &
            (stock_df['report_period'].str.endswith(('-12-31', '-03-31', '-06-30', '-09-30')))
        ].copy()

        logger.debug(f"Filtered {len(filtered_df)} quarterly/annual records for TTM calculation")
        return filtered_df

    def calculate_ttm_metrics(self, financial_df: pd.DataFrame, target_date: str) -> Dict[str, float]:
        """
        Calculate TTM metrics using correct YTD formula to avoid double counting

        TTM Formula: Current YTD + Previous Year Annual - Previous Year Same Quarter YTD

        Args:
            financial_df: Financial data DataFrame
            target_date: Target date for calculation (YYYY-MM-DD format)

        Returns:
            Dictionary of TTM metrics
        """
        if financial_df.empty:
            return self._get_empty_ttm_metrics()

        # Parse target date
        target_dt = pd.to_datetime(target_date, format='%Y-%m-%d')
        current_year = target_dt.year
        current_month = target_dt.month

        # Determine current quarter based on month
        if current_month <= 3:
            current_quarter = 1
        elif current_month <= 6:
            current_quarter = 2
        elif current_month <= 9:
            current_quarter = 3
        else:
            current_quarter = 4

        # Get TTM metrics configuration
        ttm_metrics_config = self.annual_config.get('ttm_metrics', {})

        # Initialize result dictionary with all configured metrics
        ttm_metrics = {}
        for metric_name, config in ttm_metrics_config.items():
            output_field = config.get('output_field', metric_name)
            ttm_metrics[output_field] = 0.0

        # Calculate TTM for each metric using YTD formula
        for metric_name, config in ttm_metrics_config.items():
            source_field = config.get('source_field')
            output_field = config.get('output_field', metric_name)

            if not source_field:
                continue

            ttm_value = self._calculate_single_ttm_metric(
                financial_df, source_field, current_year, current_quarter
            )

            if ttm_value is not None:
                ttm_metrics[output_field] = ttm_value
                logger.debug(".4f"
                           f"for {metric_name} (quarter {current_quarter})")

        return ttm_metrics

    def _calculate_single_ttm_metric(self, financial_df: pd.DataFrame, source_field: str,
                                   current_year: int, current_quarter: int) -> Optional[float]:
        """
        Calculate TTM for a single metric using YTD formula

        TTM = Current YTD + Previous Year Annual - Previous Year Same Quarter YTD

        Args:
            financial_df: Financial data DataFrame
            source_field: Field name to calculate TTM for
            current_year: Current year
            current_quarter: Current quarter (1-4)

        Returns:
            TTM value or None if calculation not possible
        """
        try:
            # Define quarter end dates
            quarter_ends = {
                1: '-03-31',
                2: '-06-30',
                3: '-09-30',
                4: '-12-31'
            }

            # 1. Get current year YTD value (most recent report for current quarter)
            current_ytd_value = None
            if current_quarter in quarter_ends:
                current_mask = (
                    (financial_df['report_period'].str.endswith(quarter_ends[current_quarter])) &
                    (financial_df['report_year'] == current_year)
                )
                current_data = financial_df[current_mask]
                if not current_data.empty:
                    current_ytd_value = current_data.iloc[0].get(source_field)
                    logger.debug(f"Current YTD {source_field}: {current_ytd_value} "
                               f"(Q{current_quarter} {current_year})")

            # 2. Get previous year annual value
            prev_year_annual_value = None
            prev_year = current_year - 1
            annual_mask = (
                (financial_df['report_period'].str.endswith('-12-31')) &
                (financial_df['report_year'] == prev_year)
            )
            annual_data = financial_df[annual_mask]
            if not annual_data.empty:
                prev_year_annual_value = annual_data.iloc[0].get(source_field)
                logger.debug(f"Previous year annual {source_field}: {prev_year_annual_value} "
                           f"(FY {prev_year})")

            # 3. Get previous year same quarter YTD value
            prev_year_quarter_value = None
            if current_quarter in quarter_ends:
                prev_quarter_mask = (
                    (financial_df['report_period'].str.endswith(quarter_ends[current_quarter])) &
                    (financial_df['report_year'] == prev_year)
                )
                prev_quarter_data = financial_df[prev_quarter_mask]
                if not prev_quarter_data.empty:
                    prev_year_quarter_value = prev_quarter_data.iloc[0].get(source_field)
                    logger.debug(f"Previous year Q{current_quarter} {source_field}: {prev_year_quarter_value} "
                               f"(Q{current_quarter} {prev_year})")

            # Calculate TTM using the formula: Current YTD + Prev Year Annual - Prev Year Same Quarter YTD
            if (current_ytd_value is not None and
                prev_year_annual_value is not None and
                prev_year_quarter_value is not None):

                ttm_value = current_ytd_value + prev_year_annual_value - prev_year_quarter_value

                logger.debug(f"TTM calculation for {source_field}: "
                           f"{current_ytd_value} + {prev_year_annual_value} - {prev_year_quarter_value} = {ttm_value}")

                return ttm_value
            else:
                logger.debug(f"Insufficient data for TTM calculation of {source_field}: "
                           f"Current YTD: {current_ytd_value}, "
                           f"Prev Annual: {prev_year_annual_value}, "
                           f"Prev Quarter: {prev_year_quarter_value}")
                return None

        except Exception as e:
            logger.error(f"Error calculating TTM for {source_field}: {e}")
            return None

    def calculate_compound_field(self, row: pd.Series, source_fields: Union[list, set, dict], operation: str = 'subtract') -> Optional[float]:
        """
        Calculate compound field from multiple source fields

        Args:
            row: DataFrame row containing the source fields
            source_fields: List of field names or dict with operation info
            operation: Default operation ('subtract', 'add', 'multiply', 'divide')

        Returns:
            Calculated compound value or None if calculation fails
        """
        try:
            if isinstance(source_fields, dict):
                # Dict format: {"revenue": "total_revenue", "cost": "total_cogs", "operation": "subtract"}
                # Extract field names, excluding 'operation' key
                fields = [v for k, v in source_fields.items() if k != 'operation']
                if 'operation' in source_fields:
                    operation = source_fields['operation']
            else:
                # List/set format: assume subtract operation for 2 fields
                fields = list(source_fields)

            if len(fields) < 2:
                logger.warning("Need at least 2 fields for compound calculation, got %d", len(fields))
                return None

            # Get field values
            values = []
            for field in fields:
                value = row.get(field)
                if value is None or pd.isna(value):
                    logger.debug("Missing value for field %s in compound calculation", field)
                    return None
                values.append(float(value))

            # Perform calculation based on operation
            if operation == 'subtract':
                result = values[0] - sum(values[1:])
            elif operation == 'add':
                result = sum(values)
            elif operation == 'multiply':
                result = 1.0
                for val in values:
                    result *= val
            elif operation == 'divide':
                if len(values) != 2:
                    logger.warning("Divide operation requires exactly 2 values, got %d", len(values))
                    return None
                if values[1] == 0:
                    logger.warning("Division by zero in compound calculation")
                    return None
                result = values[0] / values[1]
            else:
                logger.warning("Unsupported operation: %s", operation)
                return None

            return result

        except Exception as e:
            logger.error(f"Error calculating compound field: {e}")
            return None

    def calculate_cagr(self, financial_df: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Calculate CAGR for multiple financial metrics based on configuration"""
        if financial_df.empty:
            # Return empty results for all configured metrics
            cagr_metrics = self.annual_config.get('cagr_metrics', {})
            return {metric_name: None for metric_name in cagr_metrics.keys()}

        try:
            # Get CAGR metrics configuration
            cagr_metrics_config = self.annual_config.get('cagr_metrics', {})
            cagr_results = {}

            # Process each configured CAGR metric
            for output_field, config in cagr_metrics_config.items():
                try:
                    source_field = config.get('source_field')
                    source_fields = config.get('source_fields')
                    periods = config.get('periods', 3)
                    operation = config.get('operation', 'subtract')  # Default operation for compound fields

                    if not source_field and not source_fields:
                        logger.warning("No source_field or source_fields defined for %s", output_field)
                        cagr_results[output_field] = None
                        continue

                    # Use only annual reports for all metrics (simplified approach)
                    calculation_data = financial_df[
                        financial_df['report_period'].str.endswith('-12-31')
                    ].sort_values('report_year', ascending=False)

                    # Check if we have enough data points
                    required_data_points = periods + 1  # n-year CAGR needs n+1 data points
                    if len(calculation_data) < required_data_points:
                        logger.debug("Insufficient data for %s: need %d points, got %d", output_field, required_data_points, len(calculation_data))
                        cagr_results[output_field] = None
                        continue

                    # Get start and end values
                    if source_field:
                        # Single field calculation
                        start_value = calculation_data.iloc[-1][source_field]  # Oldest available data
                        end_value = calculation_data.iloc[0][source_field]     # Newest available data
                    elif source_fields:
                        # Compound field calculation
                        start_value = self.calculate_compound_field(calculation_data.iloc[-1], source_fields, operation)
                        end_value = self.calculate_compound_field(calculation_data.iloc[0], source_fields, operation)

                        logger.debug("Compound calculation for %s: start_value=%s, end_value=%s", output_field, start_value, end_value)

                    # Validate data and calculate CAGR
                    if (start_value is not None and start_value > 0 and
                        end_value is not None and end_value > 0):

                        # CAGR formula: (end_value / start_value)^(1/n) - 1
                        cagr_value = (end_value / start_value) ** (1/periods) - 1
                        cagr_results[output_field] = cagr_value

                        logger.debug("Calculated %s: %.4f (start: %.0f, end: %.0f, periods: %d)",
                                   output_field, cagr_value, start_value, end_value, periods)
                    else:
                        cagr_results[output_field] = None
                        logger.debug("Invalid data values for %s calculation", output_field)

                except KeyError as e:
                    # Source field doesn't exist in the data
                    cagr_results[output_field] = None
                    logger.debug("Source field not found for %s: %s", output_field, str(e))
                except Exception as e:
                    # Handle calculation errors for individual metrics
                    cagr_results[output_field] = None
                    logger.warning("Error calculating %s: %s", output_field, str(e))

            return cagr_results

        except Exception as e:
            logger.warning("CAGR calculation failed: %s", str(e))
            # Return empty results for all configured metrics
            cagr_metrics_config = self.annual_config.get('cagr_metrics', {})
            return {metric_name: None for metric_name in cagr_metrics_config.keys()}

    def _get_empty_ttm_metrics(self) -> Dict[str, float]:
        """Return empty TTM metrics dictionary based on configuration"""
        ttm_metrics_config = self.annual_config.get('ttm_metrics', {})
        empty_metrics = {}

        for metric_name, config in ttm_metrics_config.items():
            output_field = config.get('output_field', metric_name)  # Default to metric_name
            empty_metrics[output_field] = 0.0

        return empty_metrics

    def update_final_table(self, updates: List[Dict]) -> None:
        """Update final_a_stock_comb_info table with calculated metrics"""
        if not updates:
            logger.info("No updates to process")
            return

        # Prepare update statements
        update_statements = []

        for update_data in updates:
            tradedate = update_data['tradedate']
            symbol = update_data['symbol']

            # Build SET clause
            set_clauses = []
            for field, value in update_data.items():
                if field in ['tradedate', 'symbol']:
                    continue
                if value is not None:
                    set_clauses.append(f"{field} = {value}")
                else:
                    set_clauses.append(f"{field} = NULL")

            if not set_clauses:
                continue

            set_clause = ", ".join(set_clauses)

            update_sql = f"""
            UPDATE final_a_stock_comb_info
            SET {set_clause}
            WHERE tradedate = '{tradedate}' AND symbol = '{symbol}'
            """

            update_statements.append(update_sql)

        # Execute updates in batches
        batch_size = 100
        for i in range(0, len(update_statements), batch_size):
            batch = update_statements[i:i + batch_size]

            try:
                with self.engine.begin() as conn:
                    for i, sql in enumerate(batch, 1):
                        logger.debug(f"Executing update statement {i}/{len(batch)}:")
                        logger.debug(f"SQL Statement:\n{sql}")
                        conn.execute(text(sql))

                logger.info(f"Processed batch {i//batch_size + 1} ({len(batch)} updates)")

            except Exception as e:
                logger.error(f"Failed to execute batch {i//batch_size + 1}: {e}")
                # Continue with next batch rather than failing completely


def update_annual_report_ttm(
    mysql_url: str = "mysql+pymysql://root:@127.0.0.1:3306/investment_data",
    start_date: str = None,
    end_date: str = None,
    stocks: List[str] = None,
    batch_size: int = 50,
    max_workers: int = 8,
):
    """
    Update annual report TTM and CAGR metrics

    Args:
        mysql_url: MySQL connection URL
        start_date: Start date in YYYY-MM-DD format (default: current date)
        end_date: End date in YYYY-MM-DD format (default: current date)
        stocks: Optional list of specific stocks to process
        batch_size: Number of stocks to process in each batch
        max_workers: Maximum number of parallel workers
    """
    try:
        calculator = TTMCalculator(mysql_url)
        calculator.process_updates_by_stock(
            start_date=start_date,
            end_date=end_date,
            stocks=stocks,
            batch_size=batch_size,
            max_workers=max_workers
        )
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise


if __name__ == "__main__":
    fire.Fire(update_annual_report_ttm)
