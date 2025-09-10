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

@pytest.mark.parametrize("values, weights, expected", [
    (np.array([1, 2, 3, 4, 5]), np.array([1, 1, 1, 1, 1]), 3.0),
    (np.array([1, 2, 3]), np.array([1, 1, 3]), 3.0),
    (np.array([5.0]), np.array([1.0]), 5.0),
    (np.array([1.0, 3.0]), np.array([1.0, 1.0]), 2.0),
    (np.array([1.0, 3.0]), np.array([3.0, 1.0]), 1.0),
    (np.array([1.0, 3.0]), np.array([1.0, 3.0]), 3.0),
    (np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 1.0]), 2.5),
    (np.array([-5.0, -2.0, 1.0, 3.0]), np.array([1.0, 1.0, 1.0, 1.0]), -0.5),
])
def test_weighted_median(values, weights, expected):
    """Test weighted_median function with various inputs"""
    result = weighted_median(values, weights)
    assert result == pytest.approx(expected)

@pytest.mark.parametrize("eval_date, expected_quarter, expected_fiscal_year", [
    ('20240101', '2024Q1', '2023'),
    ('20240315', '2024Q1', '2023'),
    ('20240401', '2024Q2', '2024'),
    ('20240615', '2024Q2', '2024'),
    ('20240701', '2024Q3', '2024'),
    ('20240915', '2024Q3', '2024'),
    ('20241001', '2024Q4', '2024'),
    ('20241231', '2024Q4', '2024'),
])
def test_fiscal_period_info(eval_date, expected_quarter, expected_fiscal_year):
    """Test fiscal period info calculation comprehensively"""
    fiscal_info = get_fiscal_period_info(eval_date)
    assert fiscal_info['current_quarter'] == expected_quarter
    assert fiscal_info['current_fiscal_year'] == expected_fiscal_year

def test_aggregate_forecasts(mock_data):
    """Test aggregate_forecasts function with test data"""
    test_df = pd.DataFrame({
        'eps': [2.5, 2.6, 2.7, None, 2.4],
        'pe': [4.0, 4.1, 4.2, None, 3.9],
        'rd': [5.0, 5.5, None, 6.0, 4.8],
        'roe': [9.5, 10.0, 9.8, None, 9.7],
        'ev_ebitda': [8.5, 8.8, 9.0, None, 8.2],
        'max_price': [12.5, 13.0, None, 11.8, 12.2],
        'min_price': [11.5, 12.0, None, 11.0, 11.8],
        'report_type': ['点评', '一般', '点评', '一般', '非个股'],
        'rating': ['买入', '增持', '买入', '中性', '买入'],
        'quarter_comparison': [True, True, True, True, True]
    })

    with patch('evaluate_brokerage_report.get_report_weight') as mock_get_weight:
        mock_get_weight.side_effect = lambda rt: {'点评': 3.0, '一般': 2.0, '非个股': 1.0}.get(rt, 2.0)

        result_bullish = aggregate_forecasts(test_df, 'bullish')
        assert result_bullish['eps'] == pytest.approx(2.6, 0.1)
        assert result_bullish['pe'] == pytest.approx(4.05, 0.1)
        assert result_bullish['rd'] == pytest.approx(5.25, 0.1)
        assert result_bullish['roe'] == pytest.approx(9.75, 0.1)
        assert result_bullish['ev_ebitda'] == pytest.approx(8.75, 0.1)
        assert result_bullish['max_price'] == pytest.approx(12.35, 0.1)
        assert result_bullish['min_price'] == pytest.approx(11.75, 0.1)

        result_bearish = aggregate_forecasts(test_df, 'bearish')
        assert result_bearish['eps'] == pytest.approx(2.6, 0.1)  # Similar logic, adjust if needed

@pytest.mark.parametrize("ts_code, eval_date", [
    ('', '20250101'),
    (None, '20250101')
])
def test_get_brokerage_consensus_error_handling(mock_engine, ts_code, eval_date):
    """Test get_brokerage_consensus error handling"""
    result = get_brokerage_consensus(mock_engine, ts_code, eval_date, '2024')
    assert result is None

@pytest.mark.parametrize("report_type, expected_weight", [
    ('深度', 5.0),
    ('调研', 4.0),
    ('点评', 3.0),
    ('会议纪要', 3.0),
    ('一般', 2.0),
    ('新股', 1.5),
    ('港股', 1.5),
    ('非个股', 1.0),
    ('unknown_type', DEFAULT_REPORT_WEIGHT),
    (None, DEFAULT_REPORT_WEIGHT),
    ('', DEFAULT_REPORT_WEIGHT),
    ('  ', DEFAULT_REPORT_WEIGHT),
    (123, DEFAULT_REPORT_WEIGHT),
    ('深度@#$%', 5.0),
])
def test_get_report_weight(report_type, expected_weight):
    """Test get_report_weight with various inputs"""
    weight = get_report_weight(report_type)
    assert weight == expected_weight

@pytest.mark.parametrize("quarter_str, expected", [
    ('2024Q1', (2024, 1)),
    ('2023Q2', (2023, 2)),
    ('2025Q3', (2025, 3)),
    ('2022Q4', (2022, 4)),
    ('invalid', (0, 0)),
    ('', (0, 0)),
    (None, (0, 0)),
])
def test_parse_quarter(quarter_str, expected):
    """Test parse_quarter function"""
    result = parse_quarter(quarter_str)
    assert result == expected

@pytest.mark.parametrize("quarter1, quarter2, expected", [
    ('2024Q1', '2024Q1', 0),
    ('2024Q1', '2024Q2', -1),
    ('2024Q3', '2024Q2', 1),
    ('2023Q4', '2024Q1', -1),
    ('2024Q1', '2023Q4', 1),
])
def test_compare_quarters(quarter1, quarter2, expected):
    """Test compare_quarters function"""
    result = compare_quarters(quarter1, quarter2)
    assert result == expected

@pytest.mark.parametrize("report_type, expected_category", [
    ('深度报告', 'depth'),
    ('深度分析', 'depth'),
    ('调研报告', 'research'),
    ('调研纪要', 'research'),
    ('点评', 'commentary'),
    ('点评报告', 'commentary'),
    ('一般报告', 'general'),
    ('unknown', 'other'),
    (None, 'other'),
    ('', 'other'),
    ('   ', 'other'),
])
def test_categorize_report_type(report_type, expected_category):
    """Test categorize_report_type function"""
    result = categorize_report_type(report_type)
    assert result == expected_category

@pytest.mark.parametrize("rating, expected", [
    ('买入', 'BUY'),
    ('推荐', 'BUY'),
    ('增持', 'BUY'),
    ('持有', 'HOLD'),
    ('区间操作', 'HOLD'),
    ('中性', 'NEUTRAL'),
    ('Neutral', 'NEUTRAL'),
    ('卖出', 'SELL'),
    (None, 'NEUTRAL'),
    ('', 'NEUTRAL'),
    ('   ', 'NEUTRAL'),
    ('完全未知的评级', 'NEUTRAL'),
])
def test_classify_rating(rating, expected):
    """Test classify_rating function"""
    result = classify_rating(rating)
    assert result == expected

@pytest.mark.parametrize("values, weights, expected_length", [
    (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([1.0]*5), 5),
    (np.array([10.0]*20 + [1000.0]), np.array([1.0]*21), 20),
])
def test_filter_outliers(values, weights, expected_length):
    """Test _filter_outliers function"""
    filtered_values, filtered_weights = _filter_outliers(values, weights)
    assert len(filtered_values) == expected_length

@pytest.mark.parametrize("field, values, weights, expected_length", [
    ('eps', np.array([-60.0, -40.0, 0.0, 40.0, 60.0]), np.array([1.0]*5), 3),
    ('pe', np.array([-1.0, 0.0, 100.0, 600.0]), np.array([1.0]*4), 1),
])
def test_apply_field_ranges(field, values, weights, expected_length):
    """Test _apply_field_ranges function"""
    filtered_values, filtered_weights = _apply_field_ranges(field, values, weights)
    assert len(filtered_values) == expected_length

def test_aggregate_forecasts_empty_dataframe():
    """Test aggregate_forecasts with empty DataFrame"""
    empty_df = pd.DataFrame()
    result = aggregate_forecasts(empty_df, 'bullish')
    assert result['eps'] is None
    assert result['pe'] is None
    assert result['rd'] is None
    assert result['roe'] is None
    assert result['ev_ebitda'] is None
    assert result['max_price'] is None
    assert result['min_price'] is None

def test_aggregate_forecasts_missing_columns():
    """Test aggregate_forecasts with missing columns"""
    df = pd.DataFrame({
        'eps': [2.5, 2.6],
        'report_type': ['点评', '一般'],
        'report_weight': [3.0, 2.0]
    })
    result = aggregate_forecasts(df, 'bullish')
    assert result['eps'] == pytest.approx(2.55, 0.1)
    assert result['pe'] is None
    assert result['rd'] is None
    assert result['roe'] is None
    assert result['ev_ebitda'] is None
    assert result['max_price'] is None
    assert result['min_price'] is None

def test_get_date_window_invalid_format():
    """Test get_date_window with invalid date format"""
    with pytest.raises(ValueError, match="Invalid eval_date format"):
        get_date_window('invalid_date')

def test_get_trade_cal_error_handling():
    """Test get_trade_cal error handling"""
    with patch('evaluate_brokerage_report.pro.trade_cal') as mock_trade_cal:
        mock_trade_cal.side_effect = Exception("API Error")
        result = get_trade_cal('20250101', '20250110')
        assert isinstance(result, pd.DataFrame)
        assert result.empty

# Additional tests for logging can be added using caplog
@pytest.fixture
def caplog_fixture(caplog):
    caplog.set_level(logging.DEBUG)
    return caplog

def test_get_report_weight_logging(caplog_fixture):
    """Test logging in get_report_weight"""
    get_report_weight(12345)  # Non-string input
    assert any("Error converting report_type to str" in record.message for record in caplog_fixture.records)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])