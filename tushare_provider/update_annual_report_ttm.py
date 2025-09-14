"""
TTM (Trailing Twelve Months) and CAGR Calculation Module

This module calculates TTM financial metrics and CAGR from ts_a_stock_financial_profile
and updates final_a_stock_comb_info table.

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
from sqlalchemy import create_engine, text
import logging
from typing import Dict, List, Optional, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTMCalculator:
    """TTM and CAGR Calculator for financial data"""

    def __init__(self, config_path: str = 'conf/report_configs.json', annual_config_path: str = 'conf/annual_config.json'):
        """Initialize calculator with configuration"""
        self.config = self.load_config(config_path)
        self.annual_config = self.load_annual_config(annual_config_path)
        self.engine = self.create_db_engine()
        self.report_periods = {
            'annual': '%-12-31',
            'q1': '%-03-31',
            'q2': '%-06-30',
            'q3': '%-09-30'
        }

    def load_config(self, config_path: str) -> Dict:
        """Load database configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get('database', {})
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

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

    def create_db_engine(self):
        """Create SQLAlchemy database engine"""
        db_config = self.config
        connection_string = (
            f"mysql+pymysql://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        return create_engine(
            connection_string,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )

    def get_current_quarter_info(self) -> Tuple[int, int, int]:
        """Get current quarter, month, and year information"""
        today = datetime.now()
        quarter = (today.month - 1) // 3 + 1
        month = today.month
        year = today.year
        return quarter, month, year

    def get_weight_matrix(self, current_quarter: int, current_month: int) -> Dict[str, float]:
        """
        Calculate weight matrix based on current quarter and report availability

        Returns:
            Dict with keys like 'annual_2023', 'q1_2024', 'q2_2024' etc.
        """
        current_year = datetime.now().year

        # Determine report availability based on current quarter
        annual_available = current_month > 3  # Annual reports typically available after March

        weights = {}

        if current_quarter == 1:
            if annual_available:
                # Q1: Annual report available - use annual only
                weights[f'annual_{current_year-1}'] = 1.0
            else:
                # Q1: Annual report not available - mix of previous periods
                weights[f'annual_{current_year-2}'] = 0.25
                weights[f'q1_{current_year-1}'] = 0.25
                weights[f'q2_{current_year-1}'] = 0.25
                weights[f'q3_{current_year-1}'] = 0.25

        elif current_quarter == 2:
            q2_available = current_month > 4  # Q2 reports typically available after April

            if q2_available:
                # Q2: Q2 report available - mix annual and quarterly
                weights[f'annual_{current_year-1}'] = 0.5
                weights[f'q1_{current_year}'] = 0.25
                weights[f'q2_{current_year}'] = 0.25
            else:
                # Q2: Q2 report not available - mostly annual + Q1
                weights[f'annual_{current_year-1}'] = 0.75
                weights[f'q1_{current_year}'] = 0.25

        else:  # Q3 and Q4
            # Equal weight distribution
            weights[f'annual_{current_year-1}'] = 0.25
            weights[f'q1_{current_year}'] = 0.25
            weights[f'q2_{current_year}'] = 0.25
            weights[f'q3_{current_year}'] = 0.25

        return weights

    def get_target_dates(self) -> pd.DataFrame:
        """Get dates that need TTM/CAGR updates"""
        query = """
        SELECT DISTINCT trade_date
        FROM ts_a_stock_fundamental ts_raw
        LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
        WHERE STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') > COALESCE(
            (SELECT MAX(tradedate) FROM final_a_stock_comb_info), '2008-01-01'
        )
        ORDER BY trade_date
        """

        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"Found {len(df)} target dates for update")
            return df
        except Exception as e:
            logger.error(f"Failed to get target dates: {e}")
            raise

    def get_annual_data_for_cagr(self, ts_codes: List[str], target_date: str) -> pd.DataFrame:
        """Get financial data for CAGR calculation based on configuration"""
        if not ts_codes:
            return pd.DataFrame()

        # Calculate the target year from target_date
        target_date_dt = pd.to_datetime(target_date, format='%Y%m%d')
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

        ts_codes_str = ','.join([f"'{code}'" for code in ts_codes])

        # Build the WHERE clause for report periods
        report_period_conditions = " OR ".join([
            f"financial.report_period = '{year}-12-31'" for year in years_to_include
        ])

        # Get all required fields for CAGR calculation
        cagr_metrics = self.annual_config.get('cagr_metrics', {})
        required_fields = set()

        for metric_config in cagr_metrics.values():
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

        query = f"""
        SELECT
            {select_clause}
        FROM ts_a_stock_financial_profile financial
        WHERE financial.ts_code IN ({ts_codes_str})
            AND financial.ann_date <= '{target_date}'
            AND ({report_period_conditions})
        ORDER BY financial.ts_code, financial.report_period DESC
        """

        try:
            df = pd.read_sql(query, self.engine)
            if not df.empty:
                df['ann_date'] = pd.to_datetime(df['ann_date'])
                df['report_year'] = df['report_period'].str[:4].astype(int)
            logger.info(f"Retrieved {len(df)} records for CAGR calculation (years: {', '.join(years_to_include)}, max_periods: {max_periods})")
            return df
        except Exception as e:
            logger.error(f"Failed to get annual data for CAGR: {e}")
            raise

    def get_quarterly_data_for_ttm(self, ts_codes: List[str], target_date: str) -> pd.DataFrame:
        """Get quarterly and annual financial data for TTM calculation (last 12 months only)"""
        if not ts_codes:
            return pd.DataFrame()

        # Convert target_date to proper format and calculate date range
        target_date_dt = pd.to_datetime(target_date, format='%Y%m%d')
        start_date = target_date_dt - pd.DateOffset(months=12)

        ts_codes_str = ','.join([f"'{code}'" for code in ts_codes])

        query = f"""
        SELECT
            financial.ts_code,
            financial.report_period,
            financial.ann_date,
            financial.gross_profit_margin,
            financial.operating_profit_margin,
            financial.net_profit_margin,
            financial.roe_waa,
            financial.roa,
            financial.roic,
            financial.debt_to_equity,
            financial.debt_to_assets,
            financial.current_ratio,
            financial.quick_ratio,
            financial.revenue_growth
        FROM ts_a_stock_financial_profile financial
        WHERE financial.ts_code IN ({ts_codes_str})
            AND financial.ann_date <= '{target_date}'
            AND financial.ann_date >= '{start_date.strftime('%Y-%m-%d')}'
            AND (
                financial.report_period LIKE '%-12-31' OR
                financial.report_period LIKE '%-03-31' OR
                financial.report_period LIKE '%-06-30' OR
                financial.report_period LIKE '%-09-30'
            )
        ORDER BY financial.ts_code, financial.ann_date DESC
        """

        try:
            df = pd.read_sql(query, self.engine)
            if not df.empty:
                df['ann_date'] = pd.to_datetime(df['ann_date'])
                df['report_year'] = df['report_period'].str[:4].astype(int)
                df['report_quarter'] = df['report_period'].str[-5:-3]
            logger.info(f"Retrieved {len(df)} quarterly/annual records for TTM calculation")
            return df
        except Exception as e:
            logger.error(f"Failed to get quarterly data for TTM: {e}")
            raise

    def calculate_ttm_metrics(self, financial_df: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate TTM metrics using weighted approach"""
        if financial_df.empty:
            return self._get_empty_ttm_metrics()

        # Initialize result dictionary
        ttm_metrics = {
            'gross_margin_ttm': 0.0,
            'operating_margin_ttm': 0.0,
            'net_margin_ttm': 0.0,
            'roe_ttm': 0.0,
            'roa_ttm': 0.0,
            'roic_ttm': 0.0,
            'debt_to_equity_ttm': 0.0,
            'debt_to_assets_ttm': 0.0,
            'current_ratio_ttm': 0.0,
            'quick_ratio_ttm': 0.0
        }

        # Calculate weighted values for each metric
        for weight_key, weight_value in weights.items():
            period_type, period_year = weight_key.split('_')
            period_year = int(period_year)

            # Filter data for this period
            if period_type == 'annual':
                period_mask = (
                    (financial_df['report_period'].str.endswith('-12-31')) &
                    (financial_df['report_year'] == period_year)
                )
            else:
                quarter_map = {'q1': '-03-31', 'q2': '-06-30', 'q3': '-09-30'}
                period_mask = (
                    (financial_df['report_period'].str.endswith(quarter_map[period_type])) &
                    (financial_df['report_year'] == period_year)
                )

            period_data = financial_df[period_mask]
            if period_data.empty:
                continue

            # Use the most recent data for this period
            latest_data = period_data.iloc[0]

            # Add weighted values to TTM metrics
            ttm_metrics['gross_margin_ttm'] += latest_data.get('gross_profit_margin', 0) * weight_value
            ttm_metrics['operating_margin_ttm'] += latest_data.get('operating_profit_margin', 0) * weight_value
            ttm_metrics['net_margin_ttm'] += latest_data.get('net_profit_margin', 0) * weight_value
            ttm_metrics['roe_ttm'] += latest_data.get('roe_waa', 0) * weight_value
            ttm_metrics['roa_ttm'] += latest_data.get('roa', 0) * weight_value
            ttm_metrics['roic_ttm'] += latest_data.get('roic', 0) * weight_value
            ttm_metrics['debt_to_equity_ttm'] += latest_data.get('debt_to_equity', 0) * weight_value
            ttm_metrics['debt_to_assets_ttm'] += latest_data.get('debt_to_assets', 0) * weight_value
            ttm_metrics['current_ratio_ttm'] += latest_data.get('current_ratio', 0) * weight_value
            ttm_metrics['quick_ratio_ttm'] += latest_data.get('quick_ratio', 0) * weight_value

        return ttm_metrics

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
                    periods = config.get('periods', 3)

                    if not source_field:
                        logger.warning(f"No source_field defined for {output_field}")
                        cagr_results[output_field] = None
                        continue

                    # Use only annual reports for all metrics (simplified approach)
                    calculation_data = financial_df[
                        financial_df['report_period'].str.endswith('-12-31')
                    ].sort_values('report_year', ascending=False)

                    # Check if we have enough data points
                    required_data_points = periods + 1  # n-year CAGR needs n+1 data points
                    if len(calculation_data) < required_data_points:
                        logger.debug(f"Insufficient data for {output_field}: need {required_data_points} points, got {len(calculation_data)}")
                        cagr_results[output_field] = None
                        continue

                    # Get start and end values
                    start_value = calculation_data.iloc[-1][source_field]  # Oldest available data
                    end_value = calculation_data.iloc[0][source_field]     # Newest available data

                    # Validate data and calculate CAGR
                    if (start_value is not None and start_value > 0 and
                        end_value is not None and end_value > 0):

                        # CAGR formula: (end_value / start_value)^(1/n) - 1
                        cagr_value = (end_value / start_value) ** (1/periods) - 1
                        cagr_results[output_field] = cagr_value

                        logger.debug(f"Calculated {output_field}: {cagr_value:.4f} "
                                   f"(start: {start_value:.0f}, end: {end_value:.0f}, periods: {periods})")
                    else:
                        cagr_results[output_field] = None
                        logger.debug(f"Invalid data values for {output_field} calculation")

                except KeyError as e:
                    # Source field doesn't exist in the data
                    cagr_results[output_field] = None
                    logger.debug(f"Source field not found for {output_field}: {e}")
                except Exception as e:
                    # Handle calculation errors for individual metrics
                    cagr_results[output_field] = None
                    logger.warning(f"Error calculating {output_field}: {e}")

            return cagr_results

        except Exception as e:
            logger.warning(f"CAGR calculation failed: {e}")
            # Return empty results for all configured metrics
            cagr_metrics_config = self.annual_config.get('cagr_metrics', {})
            return {metric_name: None for metric_name in cagr_metrics_config.keys()}

    def _get_empty_ttm_metrics(self) -> Dict[str, float]:
        """Return empty TTM metrics dictionary"""
        return {
            'gross_margin_ttm': 0.0,
            'operating_margin_ttm': 0.0,
            'net_margin_ttm': 0.0,
            'roe_ttm': 0.0,
            'roa_ttm': 0.0,
            'roic_ttm': 0.0,
            'debt_to_equity_ttm': 0.0,
            'debt_to_assets_ttm': 0.0,
            'current_ratio_ttm': 0.0,
            'quick_ratio_ttm': 0.0
        }

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
                    for sql in batch:
                        conn.execute(text(sql))

                logger.info(f"Processed batch {i//batch_size + 1} ({len(batch)} updates)")

            except Exception as e:
                logger.error(f"Failed to execute batch {i//batch_size + 1}: {e}")
                # Continue with next batch rather than failing completely

    def process_updates(self) -> None:
        """Main processing function"""
        try:
            # Get target dates
            target_dates_df = self.get_target_dates()
            if target_dates_df.empty:
                logger.info("No dates need updating")
                return

            # Get current quarter info and weights
            current_quarter, current_month, current_year = self.get_current_quarter_info()
            weights = self.get_weight_matrix(current_quarter, current_month)

            logger.info(f"Current quarter: {current_quarter}, weights: {weights}")

            # Get all unique symbols for these dates
            symbols_query = f"""
            SELECT DISTINCT ts_link_table.w_symbol as symbol
            FROM ts_a_stock_fundamental ts_raw
            LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
            WHERE ts_raw.trade_date IN ({','.join([f"'{date}'" for date in target_dates_df['trade_date']])})
            """

            symbols_df = pd.read_sql(symbols_query, self.engine)
            symbols = symbols_df['symbol'].tolist()

            logger.info(f"Processing {len(symbols)} symbols across {len(target_dates_df)} dates")

            # Process in batches to avoid memory issues
            batch_size = 50
            all_updates = []

            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: symbols {i+1}-{min(i+batch_size, len(symbols))}")

                # Get financial data for this batch - separate queries for CAGR and TTM
                for _, date_row in target_dates_df.iterrows():
                    target_date = date_row['trade_date']

                    # Get annual data for CAGR calculation
                    annual_df = self.get_annual_data_for_cagr(batch_symbols, target_date)

                    # Get quarterly data for TTM calculation
                    quarterly_df = self.get_quarterly_data_for_ttm(batch_symbols, target_date)

                    # Process each symbol
                    for symbol in batch_symbols:
                        # Filter data for this specific symbol
                        symbol_annual_data = annual_df[annual_df['ts_code'].str.contains(symbol.split('.')[0])] if not annual_df.empty else pd.DataFrame()
                        symbol_quarterly_data = quarterly_df[quarterly_df['ts_code'].str.contains(symbol.split('.')[0])] if not quarterly_df.empty else pd.DataFrame()

                        # Calculate TTM metrics using quarterly data
                        ttm_metrics = self.calculate_ttm_metrics(symbol_quarterly_data, weights)

                        # Calculate CAGR using annual data
                        cagr_metrics = self.calculate_cagr(symbol_annual_data)

                        # Combine all metrics
                        update_data = {
                            'tradedate': target_date,
                            'symbol': symbol,
                            **ttm_metrics,
                            **cagr_metrics
                        }

                        all_updates.append(update_data)

            # Update database
            logger.info(f"Generated {len(all_updates)} updates")
            self.update_final_table(all_updates)

            logger.info("TTM and CAGR update completed successfully")

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

def main():
    """Main entry point"""
    try:
        calculator = TTMCalculator()
        calculator.process_updates()
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise

if __name__ == "__main__":
    main()
