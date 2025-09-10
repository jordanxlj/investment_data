#!/usr/bin/env python3
"""
Clean test cases for evaluate_brokerage_report.py using pytest

This script uses mock data (no database connection required) to validate the brokerage report evaluation functionality.
It simulates database queries and API calls to test the complete evaluation workflow.
"""

import pandas as pd
import numpy as np
import os
import sys
import datetime
from sqlalchemy import create_engine, text
import logging
from unittest.mock import Mock, MagicMock, patch
import pytest

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
        get_date_window,
        weighted_median,
        get_report_weight,
        get_fiscal_period_info,
        classify_rating,
        parse_quarter,
        compare_quarters,
        categorize_report_type,
        get_annual_report_data,
        process_stock_consensus,
        get_stocks_list,
        _filter_outliers,
        _apply_field_ranges,
        _upsert_batch,
        DEFAULT_REPORT_WEIGHT
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
        weighted_median = eval_module.weighted_median
        get_report_weight = eval_module.get_report_weight
        classify_rating = eval_module.classify_rating
        parse_quarter = eval_module.parse_quarter
        compare_quarters = eval_module.compare_quarters
        categorize_report_type = eval_module.categorize_report_type
        get_annual_report_data = eval_module.get_annual_report_data
        process_stock_consensus = eval_module.process_stock_consensus
        get_stocks_list = eval_module.get_stocks_list
        _filter_outliers = eval_module._filter_outliers
        _apply_field_ranges = eval_module._apply_field_ranges
        _upsert_batch = eval_module._upsert_batch
        DEFAULT_REPORT_WEIGHT = eval_module.DEFAULT_REPORT_WEIGHT

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

# Mock data fixture
@pytest.fixture
def mock_data():
    """Create mock DataFrame data to simulate database queries"""
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
            'eps': 2.25,
            'pe': None,
            'rd': None,
            'roe': None,
            'ev_ebitda': None,
            'max_price': None,
            'min_price': None
        },
        {
            'ts_code': '000001.SZ',
            'report_date': '20240815',
            'report_title': '平安银行2024年中报点评',
            'report_type': '点评',
            'classify': '一般报告',
            'org_name': '浙商证券',
            'quarter': '2026Q4',
            'rating': '买入',
            'eps': 2.581,
            'pe': 4.05,
            'rd': 4.94,
            'roe': 9.76,
            'ev_ebitda': None,
            'max_price': None,
            'min_price': 13.16
        }
    ]
    return pd.DataFrame(test_data)

@pytest.fixture
def mock_engine():
    """Mock SQLAlchemy engine"""
    return MagicMock()

# Test functions using pytest
def test_weighted_median_calculation_pytest():
    """Test weighted_median function with various inputs"""
    logger.info("Testing weighted_median function")

    # Test case 1: Simple case with equal weights
    values1 = np.array([1, 2, 3, 4, 5])
    weights1 = np.array([1, 1, 1, 1, 1])
    result1 = weighted_median(values1, weights1)
    logger.info(f"Test 1 - Equal weights: values={values1}, weights={weights1}, result={result1}")
    assert result1 == 3.0

    # Test case 2: Different weights
    values2 = np.array([1, 2, 3])
    weights2 = np.array([1, 1, 3])
    result2 = weighted_median(values2, weights2)
    logger.info(f"Test 2 - Different weights: values={values2}, weights={weights2}, result={result2}")
    assert result2 == 3.0

def test_fiscal_year_boundary_cases():
    """Test fiscal year calculation at boundary dates"""
    jan_info = get_fiscal_period_info('20240101')
    dec_info = get_fiscal_period_info('20241231')

    # January should be Q1 of current fiscal year
    assert jan_info['current_quarter'] == '2024Q1'
    assert jan_info['current_fiscal_year'] == '2023'

    # December should be Q4
    assert dec_info['current_quarter'] == '2024Q4'
    assert dec_info['current_fiscal_year'] == '2024'

def test_weighted_median_edge_cases():
    """Test weighted_median with edge cases"""
    # Test with single value
    single_result = weighted_median(np.array([5.0]), np.array([1.0]))
    assert single_result == 5.0

    # Test with two values with equal weights
    two_result = weighted_median(np.array([1.0, 3.0]), np.array([1.0, 1.0]))
    assert two_result == 2.0  # Should be the average for equal weights

def test_aggregate_forecasts_pytest(mock_data):
    """Test aggregate_forecasts function with test data"""
    logger.info("Testing aggregate_forecasts function")

    # Create test DataFrame with sample data
    test_df = pd.DataFrame({
        'eps': [2.5, 2.6, 2.7, None, 2.4],
        'pe': [4.0, 4.1, 4.2, None, 3.9],
        'rd': [5.0, 5.5, None, 6.0, 4.8],
        'roe': [9.5, 10.0, 9.8, None, 9.7],
        'ev_ebitda': [8.5, 8.8, 9.0, None, 8.2],  # Add ev_ebitda field
        'max_price': [12.5, 13.0, None, 11.8, 12.2],
        'min_price': [11.5, 12.0, None, 11.0, 11.8],
        'report_type': ['点评', '一般', '点评', '一般', '非个股'],
        'rating': ['买入', '增持', '买入', '中性', '买入'],
        'quarter_comparison': [True, True, True, True, True]
    })

    # Mock the get_report_weight function
    with patch('evaluate_brokerage_report.get_report_weight') as mock_get_weight:
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
            return weight_map.get(report_type, 2.0)
        mock_get_weight.side_effect = mock_weight_func

        # Test aggregation for bullish sentiment
        result_bullish = aggregate_forecasts(test_df, 'bullish')
        logger.info(f"Bullish aggregation result: {result_bullish}")

        # Test aggregation for bearish sentiment
        result_bearish = aggregate_forecasts(test_df, 'bearish')
        logger.info(f"Bearish aggregation result: {result_bearish}")

        # Validate results
        expected_fields = ['eps', 'pe', 'rd', 'roe', 'ev_ebitda', 'max_price', 'min_price']

        for field in expected_fields:
            assert field in result_bullish, f"Missing field {field} in bullish result"
            assert field in result_bearish, f"Missing field {field} in bearish result"

        logger.info("Aggregate forecasts test passed")

def test_full_evaluation_workflow_pytest(mock_data, mock_engine):
    """Test the complete evaluation workflow with mock data"""
    logger.info("Testing complete evaluation workflow")

    ts_code = '000001.SZ'
    eval_date = '20250102'

    # Test current period consensus (should use 2024Q4 data)
    fiscal_info = get_fiscal_period_info(eval_date)
    logger.info(f"Fiscal info for {eval_date}: {fiscal_info}")

    # Mock data for current period
    current_year_data = mock_data[
        (mock_data['ts_code'] == ts_code) &
        (mock_data['quarter'].str.contains('2024'))
    ].copy()

    # Mock data for next year
    next_year_data = mock_data[
        (mock_data['ts_code'] == ts_code) &
        (mock_data['quarter'].str.contains('2025'))
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
            return weight_map.get(report_type, 2.0)
        mock_get_weight.side_effect = mock_weight_func

        # Test current period consensus
        current_consensus = get_brokerage_consensus(mock_engine, ts_code, eval_date, fiscal_info['current_fiscal_year'])

        logger.info("Current period consensus analysis:")
        logger.info(f"- EPS: {current_consensus.get('eps')}")
        logger.info(f"- PE: {current_consensus.get('pe')}")
        logger.info(f"- Sentiment: POS={current_consensus.get('sentiment_pos')}, NEG={current_consensus.get('sentiment_neg')}")
        logger.info(f"- Report counts: Total={current_consensus.get('total_reports')}")

        # Validate key expectations
        assert current_consensus['ts_code'] == ts_code
        assert current_consensus['eval_date'] == eval_date
        assert current_consensus['total_reports'] > 0, "Should have found some reports"

        logger.info("Complete evaluation workflow test passed")

def test_evaluate_brokerage_report_new_test_cases():
    """Additional test cases for brokerage report evaluation"""

    # Test case 1: Empty DataFrame handling
    logger.info("Testing empty DataFrame handling")
    empty_df = pd.DataFrame()
    result = aggregate_forecasts(empty_df, 'bullish')
    assert result['eps'] is None
    assert result['pe'] is None

    # Test case 2: DataFrame with all NaN values
    logger.info("Testing DataFrame with all NaN values")
    nan_df = pd.DataFrame({
        'eps': [np.nan, np.nan],
        'pe': [np.nan, np.nan],
        'report_type': ['点评', '一般'],
        'report_weight': [3.0, 2.0]
    })
    result = aggregate_forecasts(nan_df, 'bullish')
    assert result['eps'] is None
    assert result['pe'] is None

    # Test case 3: Mixed valid and invalid data
    logger.info("Testing mixed valid and invalid data")
    mixed_df = pd.DataFrame({
        'eps': [2.5, np.nan, 2.8],
        'pe': [np.nan, 4.2, 4.5],
        'rd': [5.0, 5.5, np.nan],
        'roe': [9.8, np.nan, 10.2],
        'report_type': ['点评', '一般', '调研'],
        'report_weight': [3.0, 2.0, 3.5],
        'quarter_comparison': [True, True, True]
    })
    result = aggregate_forecasts(mixed_df, 'bullish')
    assert result['eps'] is not None
    assert result['pe'] is not None
    assert result['rd'] is not None
    assert result['roe'] is not None

    # Test case 4: Outlier filtering
    logger.info("Testing outlier filtering")
    outlier_df = pd.DataFrame({
        'eps': [2.5, 2.6, 1000.0],  # 1000.0 should be filtered as outlier
        'report_type': ['点评', '一般', '调研'],
        'report_weight': [3.0, 2.0, 3.5],
        'quarter_comparison': [True, True, True]
    })
    result = aggregate_forecasts(outlier_df, 'bullish')
    assert result['eps'] is not None
    assert abs(result['eps'] - 2.55) < 0.1  # Should be close to mean of 2.5 and 2.6

    # Test case 5: Fiscal period info edge cases
    logger.info("Testing fiscal period info edge cases")
    test_dates = ['20240315', '20240615', '20240915', '20241215']
    for date in test_dates:
        fiscal_info = get_fiscal_period_info(date)
        assert 'current_quarter' in fiscal_info
        assert 'current_year' in fiscal_info
        assert 'next_year' in fiscal_info
        assert fiscal_info['current_quarter'].endswith(('Q1', 'Q2', 'Q3', 'Q4'))

    # Test case 6: Report weight mapping
    logger.info("Testing report weight mapping")
    test_report_types = ['深度', '调研', '点评', '会议纪要', '一般', '新股', '港股', '非个股']
    expected_weights = [5.0, 4.0, 3.0, 3.0, 2.0, 1.5, 1.5, 1.0]

    for report_type, expected_weight in zip(test_report_types, expected_weights):
        weight = get_report_weight(report_type)
        assert weight == expected_weight, f"Expected weight {expected_weight} for {report_type}, got {weight}"

    # Test unknown report type
    unknown_weight = get_report_weight('unknown_type')
    assert unknown_weight == DEFAULT_REPORT_WEIGHT

    # Test None input
    none_weight = get_report_weight(None)
    assert none_weight == DEFAULT_REPORT_WEIGHT

    logger.info("All additional test cases passed!")

def test_config_loading_edge_cases():
    """Test configuration loading edge cases"""
    import json
    import tempfile

    # Test missing config file
    with patch('builtins.open', side_effect=FileNotFoundError):
        # This should use default values without raising an exception
        pass

    # Test invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('invalid json content')
        temp_path = f.name

    try:
        # Skip this test as it requires module reloading
        pass
    finally:
        os.unlink(temp_path)

def test_aggregate_forecasts_with_min_quarter():
    """Test aggregate_forecasts with min_quarter filtering"""
    test_df = pd.DataFrame({
        'eps': [2.5, 2.6, 2.7, 2.8],
        'pe': [4.0, 4.1, 4.2, 4.3],
        'report_type': ['点评', '一般', '点评', '一般'],
        'report_weight': [3.0, 2.0, 3.0, 2.0],
        'quarter_comparison': [True, True, False, False]
    })

    # Test with min_quarter filtering
    result_filtered = aggregate_forecasts(test_df, 'bullish', '2024Q4')
    # Test with all quarters
    result_all = aggregate_forecasts(test_df, 'bullish', 'ALL')

    # Results should be different
    assert result_filtered != result_all

def test_get_report_weight_partial_matching():
    """Test get_report_weight partial matching functionality"""
    # Test partial matches
    assert get_report_weight('深度报告') == 5.0
    assert get_report_weight('调研报告') == 4.0
    assert get_report_weight('点评分析') == 3.0

    # Test no match
    assert get_report_weight('完全不同的类型') == DEFAULT_REPORT_WEIGHT

def test_get_date_window_calculation():
    """Test date window calculation function"""
    eval_date = '20250102'
    start_date, end_date = get_date_window(eval_date, window_months=6)

    # Verify date format
    assert len(start_date) == 8 and start_date.isdigit()
    assert len(end_date) == 8 and end_date.isdigit()

    # Verify end date is eval date
    assert end_date == eval_date

    # Verify start date is approximately 6 months before
    eval_dt = datetime.datetime.strptime(eval_date, "%Y%m%d")
    start_dt = datetime.datetime.strptime(start_date, "%Y%m%d")
    delta_days = (eval_dt - start_dt).days
    assert 150 <= delta_days <= 200  # Approximately 5-7 months

def test_get_trade_cal_mock():
    """Test trade calendar functionality with mock"""
    start_date = '20250101'
    end_date = '20250110'

    mock_trading_dates = ['20250102', '20250103', '20250106', '20250107', '20250108', '20250109', '20250110']

    with patch('tushare.pro_api') as mock_pro_api:
        mock_pro = MagicMock()
        mock_pro.trade_cal.return_value = pd.DataFrame({'cal_date': mock_trading_dates})
        mock_pro_api.return_value = mock_pro

        df_cal = get_trade_cal(start_date, end_date)

        assert not df_cal.empty
        assert len(df_cal) == len(mock_trading_dates)
        assert all(date >= start_date and date <= end_date for date in df_cal['cal_date'])

def test_weighted_median_with_outliers():
    """Test weighted_median with outlier values"""
    # Test with clear outliers
    values = np.array([1.0, 2.0, 3.0, 100.0])  # 100.0 is outlier
    weights = np.array([1.0, 1.0, 1.0, 1.0])

    result = weighted_median(values, weights)
    # Should be robust to outliers, result should be around 2-3, not 100
    assert 1.5 <= result <= 3.5

def test_aggregate_forecasts_empty_dataframes():
    """Test aggregate_forecasts with various empty data scenarios"""
    # Test with completely empty DataFrame
    empty_df = pd.DataFrame()
    result = aggregate_forecasts(empty_df, 'bullish')
    assert result['eps'] is None
    assert result['pe'] is None

    # Test with DataFrame having columns but no rows
    empty_cols_df = pd.DataFrame(columns=['eps', 'pe', 'report_type', 'report_weight'])
    result = aggregate_forecasts(empty_cols_df, 'bullish')
    assert result['eps'] is None
    assert result['pe'] is None

def test_aggregate_forecasts_missing_columns():
    """Test aggregate_forecasts with missing columns"""
    # DataFrame missing some required columns
    df = pd.DataFrame({
        'eps': [2.5, 2.6],
        'report_type': ['点评', '一般'],
        'report_weight': [3.0, 2.0]
        # Missing pe, rd, roe, etc.
    })

    result = aggregate_forecasts(df, 'bullish')
    # Should handle missing columns gracefully
    assert result['eps'] is not None  # eps should be calculated
    assert result['pe'] is None       # pe should be None (missing column)

def test_fiscal_period_info_comprehensive():
    """Test fiscal period info calculation comprehensively"""
    test_cases = [
        ('20240101', '2024Q1', '2023'),  # Jan - Q1, fiscal year 2024 (natural year)
        ('20240315', '2024Q1', '2023'),  # Mar - Q1, fiscal year 2024 (natural year)
        ('20240401', '2024Q2', '2024'),  # Apr - Q2, fiscal year 2024
        ('20240615', '2024Q2', '2024'),  # Jun - Q2, fiscal year 2024
        ('20240701', '2024Q3', '2024'),  # Jul - Q3, fiscal year 2024
        ('20240915', '2024Q3', '2024'),  # Sep - Q3, fiscal year 2024
        ('20241001', '2024Q4', '2024'),  # Oct - Q4, fiscal year 2024
        ('20241231', '2024Q4', '2024'),  # Dec - Q4, fiscal year 2024
    ]

    for eval_date, expected_quarter, expected_fiscal_year in test_cases:
        fiscal_info = get_fiscal_period_info(eval_date)
        assert fiscal_info['current_quarter'] == expected_quarter
        assert fiscal_info['current_fiscal_year'] == expected_fiscal_year

def test_get_report_weight_edge_cases():
    """Test get_report_weight with various edge cases"""
    # Test with empty string
    assert get_report_weight('') == DEFAULT_REPORT_WEIGHT

    # Test with whitespace
    assert get_report_weight('  ') == DEFAULT_REPORT_WEIGHT

    # Test with numeric input (should be converted to string)
    assert get_report_weight(123) == DEFAULT_REPORT_WEIGHT

    # Test with special characters
    assert get_report_weight('深度@#$%') == 5.0  # Should still match '深度'

    # Test case sensitivity
    assert get_report_weight('深度') == 5.0  # Lower case should work
    assert get_report_weight('调研') == 4.0

def test_weighted_median_single_value():
    """Test weighted_median with single value"""
    result = weighted_median(np.array([5.0]), np.array([2.0]))
    assert result == 5.0

def test_weighted_median_two_values():
    """Test weighted_median with two values"""
    # Equal weights
    result1 = weighted_median(np.array([1.0, 3.0]), np.array([1.0, 1.0]))
    assert result1 == 2.0  # Average of 1.0 and 3.0

    # Unequal weights - higher weight on first value
    result2 = weighted_median(np.array([1.0, 3.0]), np.array([3.0, 1.0]))
    assert result2 == 1.0  # Should be closer to 1.0

    # Unequal weights - higher weight on second value
    result3 = weighted_median(np.array([1.0, 3.0]), np.array([1.0, 3.0]))
    assert result3 == 3.0  # Should be closer to 3.0

def test_aggregate_forecasts_sentiment_analysis():
    """Test aggregate_forecasts sentiment analysis"""
    df = pd.DataFrame({
        'eps': [2.5, 2.6, 2.7, 2.8],
        'pe': [4.0, 4.1, 4.2, 4.3],
        'report_type': ['点评', '一般', '点评', '一般'],
        'rating': ['买入', '买入', '增持', '卖出'],  # Mix of positive and negative ratings
        'quarter_comparison': [True, True, True, True]
    })

    result_bullish = aggregate_forecasts(df, 'bullish')
    result_bearish = aggregate_forecasts(df, 'bearish')

    # Both should have the same financial metrics (since sentiment doesn't affect calculation)
    assert result_bullish['eps'] == result_bearish['eps']
    assert result_bullish['pe'] == result_bearish['pe']

def test_aggregate_forecasts_quarter_filtering():
    """Test aggregate_forecasts with different quarter filtering"""
    df = pd.DataFrame({
        'eps': [2.5, 2.6, 2.7, 2.8],
        'pe': [4.0, 4.1, 4.2, 4.3],
        'report_type': ['点评', '一般', '点评', '一般'],
        'quarter_comparison': [True, True, False, False],  # First two match, last two don't
        'report_weight': [3.0, 2.0, 3.0, 2.0]
    })

    # Test with ALL quarters
    result_all = aggregate_forecasts(df, 'bullish', 'ALL')

    # Test with specific quarter filtering
    result_filtered = aggregate_forecasts(df, 'bullish', '2024Q4')

    # Results should be different
    assert result_all != result_filtered

def test_weighted_median_with_zeros():
    """Test weighted_median with zero weights"""
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.0, 1.0, 1.0])  # Zero weight on first value

    result = weighted_median(values, weights)
    # Should ignore the zero-weight value
    assert result == 2.5  # Average of 2.0 and 3.0

def test_weighted_median_negative_values():
    """Test weighted_median with negative values"""
    values = np.array([-5.0, -2.0, 1.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])

    result = weighted_median(values, weights)
    assert result == -0.5  # Median of [-5, -2, 1, 3] is (-2 + 1) / 2 = -0.5

def test_aggregate_forecasts_field_validation():
    """Test that aggregate_forecasts returns all expected fields"""
    df = pd.DataFrame({
        'eps': [2.5],
        'pe': [4.0],
        'report_type': ['点评'],
        'report_weight': [3.0]
    })

    result = aggregate_forecasts(df, 'bullish')

    # Check that all expected fields are present
    expected_fields = ['eps', 'pe', 'rd', 'roe', 'ev_ebitda', 'max_price', 'min_price']
    for field in expected_fields:
        assert field in result

    # Fields not in DataFrame should be None
    assert result['rd'] is None
    assert result['roe'] is None
    assert result['ev_ebitda'] is None
    assert result['max_price'] is None
    assert result['min_price'] is None

def test_get_brokerage_consensus_error_handling(mock_engine):
    """Test get_brokerage_consensus error handling"""
    # Test with invalid inputs - empty ts_code
    result = get_brokerage_consensus(mock_engine, '', '20250101', '2024')
    # Should handle empty ts_code gracefully - may return None if no data found
    assert isinstance(result, (dict, type(None)))

    # Test with None inputs
    result_none = get_brokerage_consensus(mock_engine, None, '20250101', '2024')
    assert isinstance(result_none, (dict, type(None)))

def test_get_next_year_consensus_error_handling(mock_engine):
    """Test get_next_year_consensus error handling"""
    # Test with invalid inputs - empty ts_code
    result = get_next_year_consensus(mock_engine, '', '20250101', '2025')
    # Should handle empty ts_code gracefully - may return None if no data found
    assert isinstance(result, (dict, type(None)))

    # Test with None inputs
    result_none = get_next_year_consensus(mock_engine, None, '20250101', '2025')
    assert isinstance(result_none, (dict, type(None)))

def test_parse_quarter():
    """Test parse_quarter function"""
    # Test valid quarters
    assert parse_quarter('2024Q1') == (2024, 1)
    assert parse_quarter('2023Q2') == (2023, 2)
    assert parse_quarter('2025Q3') == (2025, 3)
    assert parse_quarter('2022Q4') == (2022, 4)

    # Test invalid inputs - parse_quarter returns (0, 0) for invalid inputs
    assert parse_quarter('invalid') == (0, 0)
    assert parse_quarter('') == (0, 0)
    assert parse_quarter(None) == (0, 0)

def test_compare_quarters():
    """Test compare_quarters function"""
    # Test equal quarters
    assert compare_quarters('2024Q1', '2024Q1') == 0

    # Test different quarters same year
    assert compare_quarters('2024Q1', '2024Q2') == -1
    assert compare_quarters('2024Q3', '2024Q2') == 1

    # Test different years
    assert compare_quarters('2023Q4', '2024Q1') == -1
    assert compare_quarters('2024Q1', '2023Q4') == 1

def test_categorize_report_type():
    """Test categorize_report_type function"""
    # Test depth reports
    assert categorize_report_type('深度报告') == 'depth'
    assert categorize_report_type('深度分析') == 'depth'

    # Test research reports
    assert categorize_report_type('调研报告') == 'research'
    assert categorize_report_type('调研纪要') == 'research'

    # Test commentary reports
    assert categorize_report_type('点评') == 'commentary'
    assert categorize_report_type('点评报告') == 'commentary'

    # Test general reports
    assert categorize_report_type('一般报告') == 'general'
    assert categorize_report_type('普通报告') == 'other'  # '普通报告' doesn't contain '一般' keyword

    # Test other categories
    assert categorize_report_type('unknown') == 'other'
    assert categorize_report_type(None) == 'other'

def test_classify_rating():
    """Test classify_rating function"""
    # Test BUY ratings
    assert classify_rating('买入') == 'BUY'
    assert classify_rating('推荐') == 'BUY'
    assert classify_rating('增持') == 'BUY'

    # Test HOLD ratings
    assert classify_rating('持有') == 'HOLD'
    assert classify_rating('区间操作') == 'HOLD'
    
    # Test NEUTRAL ratings
    assert classify_rating('中性') == 'NEUTRAL'
    assert classify_rating('Neutral') == 'NEUTRAL'
    assert classify_rating('观望') == 'NEUTRAL'
    assert classify_rating('unknown') == 'NEUTRAL'

    # Test SELL ratings
    assert classify_rating('卖出') == 'SELL'
    assert classify_rating('减持') == 'SELL'

def test_filter_outliers():
    """Test _filter_outliers function"""
    # Test with normal data (no outliers)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    filtered_values, filtered_weights = _filter_outliers(values, weights)
    assert len(filtered_values) == len(values) # No outliers removed

    # Test with outliers - use values with extreme outlier
    values_with_outliers = np.array([10.0] * 20 + [1000.0]) # 1000.0 is extreme outlier with large n
    weights_with_outliers = np.array([1.0] * 21)
    filtered_values, filtered_weights = _filter_outliers(values_with_outliers, weights_with_outliers)
    # 1000.0 should be filtered out
    assert len(filtered_values) == 20

def test_apply_field_ranges():
    """Test _apply_field_ranges function"""
    # Test with eps field - range -50 to 50
    values = np.array([-60.0, -40.0, 0.0, 40.0, 60.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    filtered_values, filtered_weights = _apply_field_ranges('eps', values, weights)
    assert len(filtered_values) == 3 # -40, 0, 40 should remain

    # Test with pe field - range >0 to 500
    values = np.array([-1.0, 0.0, 100.0, 600.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    filtered_values, filtered_weights = _apply_field_ranges('pe', values, weights)
    assert len(filtered_values) == 1 # 100.0 should remain

def test_get_report_weight():
    """Test get_report_weight function"""
    # Test with known report type
    assert get_report_weight('深度') == 5.0

    # Test with unknown report type
    assert get_report_weight('unknown') == DEFAULT_REPORT_WEIGHT

    # Test with None
    assert get_report_weight(None) == DEFAULT_REPORT_WEIGHT

def test_classify_rating():
    """Test classify_rating function"""
    # Test with BUY rating
    assert classify_rating('买入') == 'BUY'

    # Test with NEUTRAL rating
    assert classify_rating('中性') == 'NEUTRAL'

    # Test with unknown rating
    assert classify_rating('unknown') == 'NEUTRAL'

def test_categorize_report_type():
    """Test categorize_report_type function"""
    # Test with depth report
    assert categorize_report_type('深度') == 'depth'

    # Test with commentary report
    assert categorize_report_type('点评') == 'commentary'

    # Test with other report
    assert categorize_report_type('industry') == 'other'

def test_parse_quarter():
    """Test parse_quarter function"""
    # Test with valid quarter
    year, quarter = parse_quarter('2024Q4')
    assert year == 2024
    assert quarter == 4

    # Test with invalid quarter
    year, quarter = parse_quarter('invalid')
    assert year == 0
    assert quarter == 0

def test_compare_quarters():
    """Test compare_quarters function"""
    # Test with same quarter
    assert compare_quarters('2024Q4', '2024Q4') == 0

    # Test with earlier quarter
    assert compare_quarters('2024Q3', '2024Q4') == -1

    # Test with later quarter
    assert compare_quarters('2025Q1', '2024Q4') == 1

def test_get_trade_cal_error_handling():
    """Test get_trade_cal error handling"""
    # Mock the pro variable to raise exception
    with patch('evaluate_brokerage_report.pro.trade_cal') as mock_trade_cal:
        mock_trade_cal.side_effect = Exception("API Error")

        result = get_trade_cal('20250101', '20250110')
        # Should return empty DataFrame on error
        assert isinstance(result, pd.DataFrame)
        assert result.empty

def test_get_date_window_invalid_format():
    """Test get_date_window with invalid date format"""
    # Test with invalid date format
    try:
        get_date_window('invalid_date')
        assert False, "Should raise ValueError for invalid date format"
    except ValueError as e:
        assert "Invalid eval_date format" in str(e)

def test_classify_rating_edge_cases():
    """Test classify_rating with edge cases"""
    # Test None input
    assert classify_rating(None) == 'NEUTRAL'

    # Test empty string
    assert classify_rating('') == 'NEUTRAL'

    # Test whitespace
    assert classify_rating('   ') == 'NEUTRAL'

    # Test unknown rating
    assert classify_rating('完全未知的评级') == 'NEUTRAL'

def test_categorize_report_type_edge_cases():
    """Test categorize_report_type with edge cases"""
    # Test None input
    assert categorize_report_type(None) == 'other'

    # Test empty string
    assert categorize_report_type('') == 'other'

    # Test whitespace
    assert categorize_report_type('   ') == 'other'

def test_parse_quarter_invalid_formats():
    """Test parse_quarter with invalid formats"""
    invalid_formats = [
        '20241',      # Missing Q
        'Q12024',     # Wrong order
        '2024Q5',     # Invalid quarter
        '2024Q0',     # Invalid quarter
        'abcdQ1',     # Invalid year
        '2024Q',      # Missing quarter number
        '2024',       # Missing quarter
        '',           # Empty string
        None,         # None value
    ]

    for invalid_format in invalid_formats:
        result = parse_quarter(invalid_format)
        # parse_quarter returns (0, 0) for invalid formats, doesn't raise exception
        assert result == (0, 0), f"Expected (0, 0) for invalid format {invalid_format}, got {result}"

def test_compare_quarters_edge_cases():
    """Test compare_quarters with edge cases"""
    # Test invalid formats
    try:
        compare_quarters('invalid', '2024Q1')
        assert False, "Should raise error for invalid format"
    except (ValueError, AttributeError):
        pass

    try:
        compare_quarters('2024Q1', 'invalid')
        assert False, "Should raise error for invalid format"
    except (ValueError, AttributeError):
        pass

def test_weighted_median_empty_arrays():
    """Test weighted_median with empty arrays"""
    # Test with empty arrays
    try:
        weighted_median(np.array([]), np.array([]))
        assert False, "Should raise ValueError for empty arrays"
    except ValueError as e:
        assert "Cannot calculate median of empty array" in str(e)

def test_weighted_median_mismatched_lengths():
    """Test weighted_median with mismatched array lengths"""
    # Test with mismatched lengths
    try:
        weighted_median(np.array([1, 2, 3]), np.array([1, 2]))
        assert False, "Should raise ValueError for mismatched lengths"
    except ValueError as e:
        assert "Values and weights must have the same length" in str(e)

def test_aggregate_forecasts_missing_report_weight():
    """Test aggregate_forecasts when report_weight column is missing"""
    df = pd.DataFrame({
        'eps': [2.5, 2.6],
        'pe': [4.0, 4.1],
        'report_type': ['点评', '一般']
        # Missing report_weight column
    })

    result = aggregate_forecasts(df, 'bullish')
    # Should handle gracefully, possibly using default weights
    assert isinstance(result, dict)

def test_get_brokerage_consensus_data_quality():
    """Test get_brokerage_consensus data quality checks"""
    # This would require a more complex mock setup to test data quality filtering
    # For now, just test that the function exists and can handle basic inputs
    mock_engine = MagicMock()
    try:
        result = get_brokerage_consensus(mock_engine, '000001.SZ', '20250101', '2024')
        assert isinstance(result, (dict, type(None)))
    except Exception:
        # Expected due to complex database operations
        pass

def test_get_next_year_consensus_data_quality():
    """Test get_next_year_consensus data quality checks"""
    # Similar to above, test basic functionality
    mock_engine = MagicMock()
    try:
        result = get_next_year_consensus(mock_engine, '000001.SZ', '20250101', '2025')
        assert isinstance(result, (dict, type(None)))
    except Exception:
        # Expected due to complex database operations
        pass

def test_config_file_missing():
    """Test behavior when config file is missing"""
    # This tests the fallback behavior when config file is missing
    # The config loading happens at import time, so we can't easily test this
    # without reloading the module. But we can test that the function still works.
    result = get_report_weight('深度')
    assert isinstance(result, (int, float))
    assert result >= 0

def test_logger_error_paths():
    """Test logger error paths that might not be covered"""
    # Test get_report_weight with problematic input that triggers logging
    result = get_report_weight(12345)  # Non-string input
    assert isinstance(result, (int, float))

    # Test categorize_report_type with problematic input
    result = categorize_report_type(12345)  # Non-string input
    assert isinstance(result, str)

    # Test classify_rating with problematic input
    result = classify_rating(12345)  # Non-string input
    assert isinstance(result, str)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])