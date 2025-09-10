#!/usr/bin/env python3
"""
Test cases for evaluate_brokerage_report.py

This script uses mock data (no database connection required) to validate the brokerage report evaluation functionality.
It simulates database queries and API calls to test the complete evaluation workflow.
"""

import pandas as pd
import os
import sys
import datetime
from sqlalchemy import create_engine, text
import logging
from unittest.mock import Mock, MagicMock, patch

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    # Try importing from the tushare module
    from tushare.evaluate_brokerage_report import (
        get_brokerage_consensus,
        get_next_year_consensus,
        aggregate_forecasts,
        get_trade_cal,
        get_fiscal_period_info,
        get_date_window
    )
except ImportError:
    # If that fails, try direct import from the file
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "evaluate_brokerage_report",
            os.path.join(project_root, "tushare", "evaluate_brokerage_report.py")
        )
        eval_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_module)
        sys.modules['evaluate_brokerage_report'] = eval_module

        # Import functions from the loaded module
        get_brokerage_consensus = eval_module.get_brokerage_consensus
        get_next_year_consensus = eval_module.get_next_year_consensus
        aggregate_forecasts = eval_module.aggregate_forecasts
        get_trade_cal = eval_module.get_trade_cal
        get_fiscal_period_info = eval_module.get_fiscal_period_info
        get_date_window = eval_module.get_date_window

        print("Successfully loaded evaluate_brokerage_report module directly")
    except Exception as e:
        print(f"Failed to load evaluate_brokerage_report: {e}")
        raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_evaluate_brokerage_report.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestBrokerageReportEvaluation:
    """Test class for brokerage report evaluation functionality"""

    def __init__(self, mysql_url=None):
        # Mock database connection instead of real connection
        self.mysql_url = mysql_url or "mock://test_db"
        self.engine = MagicMock()  # Mock engine
        self.mock_data = self._create_mock_data()

    def _create_mock_data(self):
        """Create mock DataFrame data to simulate database queries"""
        # Complete test data based on the provided sample
        test_data = [
            {
                'ts_code': '000001.SZ',
                'report_date': '20240601',
                'report_title': '银行业2024年投资策略',
                'report_type': '非个股',
                'classify': '一般报告',
                'org_name': '国泰君安',
                'quarter': '2024Q4',
                'rating': '增持',
                'op_rt': None,
                'op_pr': None,
                'tp': None,
                'np': 4366330,
                'eps': 2.25,
                'pe': 4.2,
                'rd': None,
                'roe': None,
                'ev_ebitda': None,
                'max_price': None,
                'min_price': 12.5
            },
            {
                'ts_code': '000001.SZ',
                'report_date': '20240721',
                'report_title': '商业银行：24Q2板块市值提升带动银行股仓位提高——银行板块资金流向跟踪报告',
                'report_type': '非个股',
                'classify': '一般报告',
                'org_name': '国泰君安',
                'quarter': '2024Q4',
                'rating': '增持',
                'op_rt': None,
                'op_pr': None,
                'tp': None,
                'np': 4463360,
                'eps': 2.3,
                'pe': 4.1,
                'rd': None,
                'roe': None,
                'ev_ebitda': None,
                'max_price': None,
                'min_price': 12.5
            },
            {
                'ts_code': '000001.SZ',
                'report_date': '20240815',
                'report_title': '平安银行2024年中报点评：业务调整曙光现',
                'report_type': '点评',
                'classify': '一般报告',
                'org_name': '浙商证券',
                'quarter': '2026Q4',
                'rating': '买入',
                'op_rt': 15315800,
                'op_pr': None,
                'tp': 6224000,
                'np': 5009500,
                'eps': 2.581,
                'pe': 4.05,
                'rd': 4.94,
                'roe': 9.76,
                'ev_ebitda': None,
                'max_price': None,
                'min_price': 13.16
            },
            {
                'ts_code': '000001.SZ',
                'report_date': '20240816',
                'report_title': '平安银行：中期分红落地，风险底线再强化',
                'report_type': '点评',
                'classify': '一般报告',
                'org_name': '中信建投',
                'quarter': '2026Q4',
                'rating': '买入',
                'op_rt': 15747700,
                'op_pr': None,
                'tp': 6341400,
                'np': 5199900,
                'eps': 2.68,
                'pe': 4.0,
                'rd': None,
                'roe': 9.66,
                'ev_ebitda': None,
                'max_price': None,
                'min_price': 14.7
            },
            {
                'ts_code': '000001.SZ',
                'report_date': '20240816',
                'report_title': '平安银行：中期分红方案公布，主动调整进行时',
                'report_type': '一般',
                'classify': '一般报告',
                'org_name': '中金',
                'quarter': '2025Q4',
                'rating': '跑赢行业',
                'op_rt': 15125400,
                'op_pr': None,
                'tp': 6060600,
                'np': 4878000,
                'eps': 2.51,
                'pe': 4.0,
                'rd': 7.5,
                'roe': 9.4,
                'ev_ebitda': None,
                'max_price': None,
                'min_price': 13.18
            },
            {
                'ts_code': '000001.SZ',
                'report_date': '20240816',
                'report_title': '平安银行2024年半年报点评：盈利维持正增，中期分红安排出炉',
                'report_type': '点评',
                'classify': '一般报告',
                'org_name': '光大证券',
                'quarter': '2026Q4',
                'rating': '买入',
                'op_rt': 17163200,
                'op_pr': None,
                'tp': 6427200,
                'np': 5173900,
                'eps': 2.67,
                'pe': 3.76,
                'rd': None,
                'roe': 10.57,
                'ev_ebitda': None,
                'max_price': None,
                'min_price': 12.5
            },
            {
                'ts_code': '000001.SZ',
                'report_date': '20240816',
                'report_title': '平安银行2024年半年报点评：中期分红方案落地，利润维持正增长',
                'report_type': '点评',
                'classify': '一般报告',
                'org_name': '兴业证券',
                'quarter': '2025Q4',
                'rating': '增持',
                'op_rt': 14810300,
                'op_pr': None,
                'tp': 5895900,
                'np': 4745400,
                'eps': 2.445,
                'pe': None,
                'rd': None,
                'roe': 10.02,
                'ev_ebitda': None,
                'max_price': None,
                'min_price': 12.5
            },
            {
                'ts_code': '000001.SZ',
                'report_date': '20240816',
                'report_title': '平安银行2024年半年报点评：结构调优，资产质量趋稳，中期分红率20%',
                'report_type': '点评',
                'classify': '一般报告',
                'org_name': '华创证券',
                'quarter': '2026Q4',
                'rating': '推荐',
                'op_rt': 14373900,
                'op_pr': None,
                'tp': 6367400,
                'np': 5125800,
                'eps': 2.641,
                'pe': 3.86,
                'rd': None,
                'roe': None,
                'ev_ebitda': None,
                'max_price': None,
                'min_price': 13.5
            },
            {
                'ts_code': '000001.SZ',
                'report_date': '20240816',
                'report_title': '平安银行：结构调整加快，中期分红落地',
                'report_type': '点评',
                'classify': '一般报告',
                'org_name': '华泰证券',
                'quarter': '2026Q4',
                'rating': '买入',
                'op_rt': 14701400,
                'op_pr': None,
                'tp': None,
                'np': 5161400,
                'eps': 2.66,
                'pe': 3.77,
                'rd': 7.96,
                'roe': 9.23,
                'ev_ebitda': None,
                'max_price': None,
                'min_price': 12.58
            },
            {
                'ts_code': '000001.SZ',
                'report_date': '20240816',
                'report_title': '平安银行2024年中报点评：非息收入增长亮眼',
                'report_type': '点评',
                'classify': '一般报告',
                'org_name': '国信证券',
                'quarter': '2025Q4',
                'rating': '中性',
                'op_rt': 14826300,
                'op_pr': None,
                'tp': 6000000,
                'np': 4838700,
                'eps': 2.493,
                'pe': 4.4,
                'rd': 7.5,
                'roe': 10.8,
                'ev_ebitda': None,
                'max_price': None,
                'min_price': 12.5
            }
        ]

        return pd.DataFrame(test_data)

    def setup_test_data(self):
        """Setup mock test data (no actual database operations)"""
        logger.info(f"Mock test data ready: {len(self.mock_data)} records")
        logger.info("Mock database setup completed")

    def test_get_brokerage_consensus(self):
        """Test get_brokerage_consensus function with mock data"""

        logger.info("Testing get_brokerage_consensus function")

        ts_code = '000001.SZ'
        eval_date = '20250102'  # Test date

        # Mock the database query result
        # Filter mock data for current fiscal year (2024) and ts_code
        current_year_data = self.mock_data[
            (self.mock_data['ts_code'] == ts_code) &
            (self.mock_data['quarter'].str.contains('2024'))
        ].copy()

        logger.info(f"Mock data for current period: {len(current_year_data)} reports")

        # Add report_weight column to mock data
        weight_map = {
            '深度': 4.0,
            '调研': 3.5,
            '点评': 3.0,
            '会议纪要': 3.0,
            '一般': 2.0,
            '新股': 1.5,
            '港股': 1.5,
            '非个股': 1.0
        }
        current_year_data = current_year_data.copy()
        current_year_data['report_weight'] = current_year_data['report_type'].map(
            lambda x: weight_map.get(str(x).strip(), 2.0) if x is not None else 2.0
        ).astype(float)

        # Mock the engine's pd.read_sql to return our test data
        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.return_value = current_year_data

            # Call the actual function
            result = get_brokerage_consensus(self.engine, ts_code, eval_date, '2024')

            logger.info(f"Current period consensus result: {result}")

            # Validate result structure
            expected_fields = [
                'ts_code', 'eval_date', 'report_period', 'eps', 'pe', 'rd', 'roe',
                'ev_ebitda', 'max_price', 'min_price', 'total_reports', 'sentiment_pos',
                'sentiment_neg', 'buy_count', 'depth_reports', 'research_reports',
                'commentary_reports', 'general_reports', 'other_reports', 'avg_report_weight',
                'data_source', 'last_updated'
            ]

            for field in expected_fields:
                assert field in result, f"Missing field: {field}"

            assert result['ts_code'] == ts_code
            assert result['eval_date'] == eval_date
            assert result['data_source'] == 'brokerage_consensus'

            logger.info("get_brokerage_consensus test passed")

    def test_get_next_year_consensus(self):
        """Test get_next_year_consensus function with mock data"""

        logger.info("Testing get_next_year_consensus function")

        ts_code = '000001.SZ'
        eval_date = '20250102'  # Test date

        # Mock the database query result
        # Filter mock data for next year (2025) and ts_code
        next_year_data = self.mock_data[
            (self.mock_data['ts_code'] == ts_code) &
            (self.mock_data['quarter'].str.contains('2025'))
        ].copy()

        logger.info(f"Mock data for next year: {len(next_year_data)} reports")

        # Add report_weight column to mock data for next year
        weight_map = {
            '深度': 4.0,
            '调研': 3.5,
            '点评': 3.0,
            '会议纪要': 3.0,
            '一般': 2.0,
            '新股': 1.5,
            '港股': 1.5,
            '非个股': 1.0
        }
        next_year_data = next_year_data.copy()
        next_year_data['report_weight'] = next_year_data['report_type'].map(
            lambda x: weight_map.get(str(x).strip(), 2.0) if x is not None else 2.0
        ).astype(float)

        # Mock the engine's pd.read_sql to return our test data
        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.return_value = next_year_data

            # Call the actual function
            result = get_next_year_consensus(self.engine, ts_code, eval_date, '2025')

            logger.info(f"Next year consensus result: {result}")

            if result:
                # Validate result structure for get_next_year_consensus
                expected_fields = [
                    'total_reports', 'avg_report_weight', 'eps', 'pe', 'roe', 'ev_ebitda'
                ]

                for field in expected_fields:
                    assert field in result, f"Missing field: {field}"

                # Validate that we have some reports
                assert result['total_reports'] > 0, "Should have found some reports"

            logger.info("get_next_year_consensus test passed")

    def test_fiscal_period_info(self):
        """Test fiscal period information calculation"""

        logger.info("Testing fiscal period information calculation")

        test_dates = ['20250102', '20250415', '20250720', '20251025']

        for eval_date in test_dates:
            fiscal_info = get_fiscal_period_info(eval_date)

            logger.info(f"Date {eval_date}: fiscal_info = {fiscal_info}")

            # Validate fiscal period logic
            eval_dt = datetime.datetime.strptime(eval_date, "%Y%m%d")
            month = eval_dt.month

            if month <= 3:
                # Q1: current should be previous year Q4, next should be current year Q4
                assert fiscal_info['current_fiscal_year'] == str(eval_dt.year - 1)
                assert fiscal_info['next_fiscal_year'] == str(eval_dt.year)
            else:
                # Other quarters: next year should be next calendar year
                assert fiscal_info['next_fiscal_year'] == str(eval_dt.year + 1)

        logger.info("Fiscal period info test passed")

    def test_date_window_calculation(self):
        """Test date window calculation"""

        logger.info("Testing date window calculation")

        eval_date = '20250102'
        start_date, end_date = get_date_window(eval_date, window_months=6)

        logger.info(f"Date window for {eval_date}: {start_date} to {end_date}")

        # Validate date window
        eval_dt = datetime.datetime.strptime(eval_date, "%Y%m%d")
        start_dt = datetime.datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.datetime.strptime(end_date, "%Y%m%d")

        # Check that window covers approximately 6 months back
        delta_days = (eval_dt - start_dt).days
        assert delta_days >= 150 and delta_days <= 200, f"Date window delta {delta_days} days is incorrect"

        assert end_dt == eval_dt, "End date should be the eval date"

        logger.info("Date window calculation test passed")

    def test_trade_calendar(self):
        """Test trade calendar functionality with mock data"""

        logger.info("Testing trade calendar functionality")

        start_date = '20250101'
        end_date = '20250110'

        # Mock trading calendar data
        mock_trading_dates = ['20250102', '20250103', '20250106', '20250107', '20250108', '20250109', '20250110']
        mock_df = pd.DataFrame({'cal_date': mock_trading_dates})

        # Mock the Tushare API call
        with patch('tushare.pro_api') as mock_pro_api:
            mock_pro = MagicMock()
            mock_pro.trade_cal.return_value = mock_df
            mock_pro_api.return_value = mock_pro

            df_cal = get_trade_cal(start_date, end_date)

            if not df_cal.empty:
                logger.info(f"Found {len(df_cal)} trading days between {start_date} and {end_date}")
                logger.info(f"Trading dates: {df_cal['cal_date'].tolist()}")

                # Validate that all dates are within the range
                for date in df_cal['cal_date']:
                    assert date >= start_date and date <= end_date, f"Date {date} is outside the range"

                # Validate we got the expected number of trading days
                assert len(df_cal) == len(mock_trading_dates), f"Expected {len(mock_trading_dates)} trading days, got {len(df_cal)}"
            else:
                logger.warning("No trading calendar data available")

        logger.info("Trade calendar test passed")

    def test_aggregate_forecasts(self):
        """Test aggregate_forecasts function with test data"""

        logger.info("Testing aggregate_forecasts function")

        # Create test DataFrame with sample data
        test_df = pd.DataFrame({
            'eps': [2.5, 2.6, 2.7, None, 2.4],
            'pe': [4.0, 4.1, 4.2, None, 3.9],
            'rd': [5.0, 5.5, None, 6.0, 4.8],
            'roe': [9.5, 10.0, 9.8, None, 9.7],
            'max_price': [12.5, 13.0, None, 11.8, 12.2],
            'min_price': [11.5, 12.0, None, 11.0, 11.8],
            'report_type': ['点评', '一般', '点评', '一般', '非个股'],  # Add report types for weighting
            'rating': ['买入', '增持', '买入', '中性', '买入'],  # Add ratings for sentiment
            'quarter_comparison': [True, True, True, True, True]  # Mock quarter comparison
        })

        # Add report_weight column directly to avoid get_report_weight issues
        weight_map = {
            '深度': 4.0,
            '调研': 3.5,
            '点评': 3.0,
            '会议纪要': 3.0,
            '一般': 2.0,
            '新股': 1.5,
            '港股': 1.5,
            '非个股': 1.0
        }
        test_df['report_weight'] = test_df['report_type'].map(
            lambda x: weight_map.get(str(x).strip(), 2.0) if x is not None else 2.0
        ).astype(float)

        # Set the min_quarter variable in the global scope for aggregate_forecasts
        import evaluate_brokerage_report as eval_module
        eval_module.min_quarter = 'ALL'  # Use 'ALL' to aggregate all quarters

        # Test aggregation for bullish sentiment
        result_bullish = aggregate_forecasts(test_df, 'bullish')

        logger.info(f"Bullish aggregation result: {result_bullish}")

        # Test aggregation for bearish sentiment
        result_bearish = aggregate_forecasts(test_df, 'bearish')

        logger.info(f"Bearish aggregation result: {result_bearish}")

        # Validate results
        expected_fields = ['eps', 'pe', 'rd', 'roe', 'ev_ebitda', 'max_price', 'min_price']

        for field in expected_fields:
            # Results should contain all expected fields
            assert field in result_bullish, f"Missing field {field} in bullish result"
            assert field in result_bearish, f"Missing field {field} in bearish result"

            # Values should be numeric when present
            if result_bullish[field] is not None:
                assert isinstance(result_bullish[field], (int, float)), f"{field} should be numeric in bullish result"
            if result_bearish[field] is not None:
                assert isinstance(result_bearish[field], (int, float)), f"{field} should be numeric in bearish result"

        # Additional validation for weighted average calculations
        self._validate_weighted_averages(test_df, result_bullish, 'bullish')

        logger.info("Aggregate forecasts test passed")

    def _validate_weighted_averages(self, df, result, sentiment_source):
        """Validate weighted median calculations with same logic as aggregate_forecasts"""
        logger.info(f"Validating weighted medians for {sentiment_source} sentiment")

        # Import required functions
        import numpy as np
        from tushare.evaluate_brokerage_report import weighted_median

        # Test fields that should have weighted medians
        test_fields = ['eps', 'pe', 'rd', 'roe', 'max_price', 'min_price']

        for field in test_fields:
            if field in df.columns and result.get(field) is not None:
                # Get valid values and weights (same as aggregate_forecasts)
                field_data = df[[field, 'report_weight']].dropna()
                if not field_data.empty:
                    values = field_data[field].values
                    weights = field_data['report_weight'].values

                    # Apply same outlier filtering as aggregate_forecasts
                    if len(values) > 2:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        valid_mask = (values >= mean_val - 3 * std_val) & (values <= mean_val + 3 * std_val)
                        values = values[valid_mask]
                        weights = weights[valid_mask]

                    # Apply same range filtering as aggregate_forecasts
                    if field == 'eps':
                        valid_mask = (values >= -50) & (values <= 50)
                    elif field in ['pe', 'ev_ebitda']:
                        valid_mask = (values > 0) & (values <= 500)
                    elif field == 'roe':
                        valid_mask = (values >= -200) & (values <= 200)
                    elif field in ['max_price', 'min_price']:
                        valid_mask = (values > 0) & (values <= 10000)
                    else:
                        valid_mask = np.ones(len(values), dtype=bool)

                    values = values[valid_mask]
                    weights = weights[valid_mask]

                    if len(values) > 0:
                        # Calculate expected weighted median using same function
                        if len(values) == 1:
                            expected_weighted_median = float(values[0])
                        else:
                            expected_weighted_median = float(weighted_median(values, weights))

                        actual_result = result[field]

                        logger.debug(f"{field}: expected weighted median={expected_weighted_median:.4f}, "
                                   f"actual result={actual_result:.4f}")

                        # The results should match exactly (since we're using the same algorithm)
                        diff = abs(actual_result - expected_weighted_median)
                        tolerance = 0.001  # Very small tolerance for exact match

                        assert diff <= tolerance, (
                            f"{field} result {actual_result:.4f} differs from "
                            f"expected weighted median {expected_weighted_median:.4f} "
                            f"(diff: {diff:.6f})"
                        )

                        # Result should be within the range of filtered input values
                        if len(values) > 0:
                            min_val, max_val = values.min(), values.max()
                            assert min_val <= actual_result <= max_val, (
                                f"{field} result {actual_result:.4f} is outside filtered input range "
                                f"[{min_val:.4f}, {max_val:.4f}]"
                            )

        logger.info("Weighted median validation completed")

    def test_weighted_median_calculation(self):
        """Test weighted median calculation with known values"""
        logger.info("Testing weighted median calculation with known values")

        # Create test data with known weighted median
        # Values: [1, 3, 5, 7, 9]
        # Weights: [1, 1, 1, 1, 1] -> Simple median should be 5
        test_data = {
            'eps': [1.0, 3.0, 5.0, 7.0, 9.0],
            'pe': [1.0, 3.0, 5.0, 7.0, 9.0],
            'report_type': ['点评', '一般', '点评', '一般', '点评'],
            'rating': ['买入', '增持', '买入', '中性', '买入'],
            'quarter_comparison': [True, True, True, True, True]
        }

        df = pd.DataFrame(test_data)

        # Add equal weights for simple case
        df['report_weight'] = 1.0

        # Set min_quarter
        import evaluate_brokerage_report as eval_module
        eval_module.min_quarter = 'ALL'

        # Test with equal weights - should get median
        result = aggregate_forecasts(df, 'bullish')

        # With equal weights and odd number of values, should get exact median
        expected_median = 5.0
        assert abs(result['eps'] - expected_median) < 0.001, (
            f"EPS result {result['eps']:.4f} should be close to median {expected_median}"
        )
        assert abs(result['pe'] - expected_median) < 0.001, (
            f"PE result {result['pe']:.4f} should be close to median {expected_median}"
        )

        logger.info("Equal weights median test passed")

        # Test with unequal weights - create a case where high weight pulls median toward high value
        # Values: [1, 10, 20, 30, 100]
        # Weights: [1, 1, 1, 1, 8] -> Total weight = 12, median weight = 6
        # The value 100 has weight 8, so median should be 100
        test_data_weighted = {
            'eps': [1.0, 10.0, 20.0, 30.0, 100.0],
            'pe': [1.0, 10.0, 20.0, 30.0, 100.0],
            'report_type': ['一般', '点评', '点评', '点评', '深度'],  # '深度' has highest weight
            'rating': ['买入', '增持', '买入', '中性', '买入'],
            'quarter_comparison': [True, True, True, True, True]
        }

        df_weighted = pd.DataFrame(test_data_weighted)
        weight_map = {'深度': 4.0, '点评': 3.0, '一般': 2.0}
        df_weighted['report_weight'] = df_weighted['report_type'].map(
            lambda x: weight_map.get(str(x).strip(), 2.0)
        ).astype(float)

        # Expected: weights = [2.0, 3.0, 3.0, 3.0, 4.0]
        # cum_weights = [2, 5, 8, 11, 15], total=15, median_weight=7.5
        # Should land on index where cum_weight >= 7.5, which would be around value 20 or 30

        result_weighted = aggregate_forecasts(df_weighted, 'bullish')

        # The weighted median should be influenced by the higher weights on larger values
        # Expected result should be around 20-30, not the low value 1.0
        assert result_weighted['eps'] >= 10.0, (
            f"EPS result {result_weighted['eps']:.4f} should be >= 10.0 (higher than lowest value)"
        )
        assert result_weighted['pe'] >= 10.0, (
            f"PE result {result_weighted['pe']:.4f} should be >= 10.0 (higher than lowest value)"
        )

        # Log the actual weights for debugging
        logger.info(f"Weighted test - report types: {df_weighted['report_type'].tolist()}")
        logger.info(f"Weighted test - weights: {df_weighted['report_weight'].tolist()}")
        logger.info(f"Weighted test - eps values: {df_weighted['eps'].tolist()}")
        logger.info(f"Weighted test - eps result: {result_weighted['eps']:.4f}")

        logger.info("Unequal weights test passed")

    def test_full_evaluation_workflow(self):
        """Test the complete evaluation workflow with mock data"""

        logger.info("Testing complete evaluation workflow")

        # Test data covers multiple scenarios:
        # - Different quarters: 2024Q4, 2025Q4, 2026Q4
        # - Different report types: 点评, 一般, 非个股
        # - Different ratings: 买入, 增持, 中性, 跑赢行业, 优于大市, etc.
        # - Mixed financial metrics: eps, pe, rd, roe, max_price, min_price

        ts_code = '000001.SZ'
        eval_date = '20250102'

        # Test current period consensus (should use 2024Q4 data)
        fiscal_info = get_fiscal_period_info(eval_date)
        logger.info(f"Fiscal info for {eval_date}: {fiscal_info}")

        # Mock data for current period
        current_year_data = self.mock_data[
            (self.mock_data['ts_code'] == ts_code) &
            (self.mock_data['quarter'].str.contains('2024'))
        ].copy()

        # Mock data for next year
        next_year_data = self.mock_data[
            (self.mock_data['ts_code'] == ts_code) &
            (self.mock_data['quarter'].str.contains('2025'))
        ].copy()

        logger.info(f"Mock data - Current period: {len(current_year_data)} reports, Next year: {len(next_year_data)} reports")

        with patch('pandas.read_sql') as mock_read_sql, \
             patch('evaluate_brokerage_report.get_report_weight') as mock_get_weight:

            # Mock current period query
            def mock_read_sql_side_effect(*args, **kwargs):
                query = args[0] if args else ""
                if '2024' in str(query):
                    return current_year_data
                elif '2025' in str(query):
                    return next_year_data
                else:
                    return current_year_data

            mock_read_sql.side_effect = mock_read_sql_side_effect

            def mock_weight_func(report_type):
                weight_map = {
                    '深度': 4.0,
                    '调研': 3.5,
                    '点评': 3.0,
                    '会议纪要': 3.0,
                    '一般': 2.0,
                    '新股': 1.5,
                    '港股': 1.5,
                    '非个股': 1.0
                }
                # Ensure we always return a float, even for None or unexpected values
                if report_type is None:
                    return 2.0
                weight = weight_map.get(str(report_type).strip(), 2.0)
                return float(weight)
            mock_get_weight.side_effect = mock_weight_func

            # Test current period consensus
            current_consensus = get_brokerage_consensus(self.engine, ts_code, eval_date, fiscal_info['current_fiscal_year'])

            logger.info("Current period consensus analysis:")
            logger.info(f"- EPS: {current_consensus.get('eps')}")
            logger.info(f"- PE: {current_consensus.get('pe')}")
            logger.info(f"- Sentiment: POS={current_consensus.get('sentiment_pos')}, NEG={current_consensus.get('sentiment_neg')}")
            logger.info(f"- Report counts: Total={current_consensus.get('total_reports')}")
            logger.info(f"- Price range: Max={current_consensus.get('max_price')}, Min={current_consensus.get('min_price')}")

            # Test next year consensus (should use 2025Q4 data for 20250102)
            next_year_consensus = get_next_year_consensus(self.engine, ts_code, eval_date, fiscal_info['next_fiscal_year'])

            if next_year_consensus:
                logger.info("Next year consensus analysis:")
                logger.info(f"- Next year EPS: {next_year_consensus.get('eps')}")
                logger.info(f"- Next year PE: {next_year_consensus.get('pe')}")
                logger.info(f"- Next year reports: {next_year_consensus.get('total_reports')}")
            else:
                logger.warning("No next year consensus data found")

            # Validate key expectations
            assert current_consensus['ts_code'] == ts_code
            assert current_consensus['eval_date'] == eval_date
            assert current_consensus['total_reports'] > 0, "Should have found some reports"

            # For 2024Q4 data (current period), we should have financial metrics
            assert current_consensus['eps'] is not None, "Should have EPS data for current period"
            assert current_consensus['pe'] is not None, "Should have PE data for current period"

            # For price data (all reports), we should have some values
            assert current_consensus['min_price'] is not None, "Should have min_price from all reports"

            logger.info("Complete evaluation workflow test passed")

    def test_quarter_specific_vs_all_report_aggregation(self):
        """Test the difference between quarter-specific and all-report aggregation"""

        logger.info("Testing quarter-specific vs all-report aggregation")

        # Create test DataFrames to simulate the aggregation logic
        import pandas as pd
        import numpy as np

        # Simulate reports with different quarters and price data
        test_data = {
            'quarter': ['2024Q4', '2024Q4', '2025Q4', '2025Q4', '2026Q4', '2026Q4'],
            'eps': [2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
            'pe': [4.0, 4.1, 4.2, 4.3, 4.4, 4.5],
            'max_price': [12.5, 13.0, 13.5, 14.0, 14.5, 15.0],
            'min_price': [11.5, 12.0, 12.5, 13.0, 13.5, 14.0],
            'report_type': ['点评', '一般', '点评', '一般', '点评', '一般'],
            'rating': ['买入', '增持', '买入', '中性', '买入', '增持'],
            'quarter_comparison': [True, True, False, False, False, False]  # Only first 2 match 2024Q4
        }

        df = pd.DataFrame(test_data)

        # Set the min_quarter variable in the global scope for aggregate_forecasts
        import evaluate_brokerage_report as eval_module
        eval_module.min_quarter = '2024Q4'  # Test quarter-specific filtering

        # Test with quarter-specific filtering (should only use first 2 rows)
        result_quarter_specific = aggregate_forecasts(df, 'bullish')
        logger.info(f"Quarter-specific aggregation (2024Q4): EPS={result_quarter_specific.get('eps')}, PE={result_quarter_specific.get('pe')}")

        # Test with all quarters (should use all rows)
        eval_module.min_quarter = 'ALL'
        result_all_quarters = aggregate_forecasts(df, 'bullish')
        logger.info(f"All quarters aggregation: Max Price={result_all_quarters.get('max_price')}, Min Price={result_all_quarters.get('min_price')}")

        # Validate that all-report aggregation considers all data
        assert result_all_quarters['max_price'] is not None
        assert result_all_quarters['min_price'] is not None

        # Validate that quarter-specific considers only relevant data
        assert result_quarter_specific['eps'] is not None
        assert result_quarter_specific['pe'] is not None

        logger.info("Quarter-specific vs all-report aggregation test passed")

    def run_all_tests(self):
        """Run all test cases"""

        logger.info("Starting test suite for evaluate_brokerage_report.py")

        try:
            # Setup test data
            self.setup_test_data()

            # Run individual tests
            self.test_fiscal_period_info()
            self.test_date_window_calculation()
            self.test_trade_calendar()
            self.test_aggregate_forecasts()
            self.test_weighted_median_calculation()
            self.test_get_brokerage_consensus()
            self.test_get_next_year_consensus()
            self.test_full_evaluation_workflow()
            self.test_quarter_specific_vs_all_report_aggregation()

            logger.info("All tests passed successfully!")

        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise

def main():
    """Main function to run tests"""

    import argparse

    parser = argparse.ArgumentParser(description='Test evaluate_brokerage_report.py functionality')
    parser.add_argument('--mysql_url', help='MySQL connection URL', default=None)
    parser.add_argument('--setup_only', action='store_true', help='Only setup test data, do not run tests')

    args = parser.parse_args()

    tester = TestBrokerageReportEvaluation(args.mysql_url)

    if args.setup_only:
        tester.setup_test_data()
        logger.info("Test data setup completed")
    else:
        tester.run_all_tests()

if __name__ == '__main__':
    main()
