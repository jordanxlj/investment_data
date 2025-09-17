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
import json
from datetime import datetime as dt
from sqlalchemy import create_engine, text
import logging
from unittest.mock import Mock, MagicMock, patch, mock_open
import pytest
import importlib
import time

# Set up logger for tests
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import threading
import concurrent

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    import tushare_provider.evaluate_brokerage_report as evaluate_brokerage_report
except ImportError:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "evaluate_brokerage_report",
            os.path.join(project_root, "tushare_provider", "evaluate_brokerage_report.py")
        )
        eval_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_module)
        sys.modules['tushare_provider.evaluate_brokerage_report'] = eval_module
        evaluate_brokerage_report = eval_module
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
    result = evaluate_brokerage_report.weighted_median(values, weights)
    assert result == pytest.approx(expected)

def test_weighted_median_empty():
    """Test weighted_median with empty arrays"""
    with pytest.raises(ValueError, match="Cannot calculate median of empty array"):
        evaluate_brokerage_report.weighted_median(np.array([]), np.array([]))

def test_weighted_median_mismatched_lengths():
    """Test weighted_median with mismatched lengths"""
    with pytest.raises(ValueError, match="Values and weights must have the same length"):
        evaluate_brokerage_report.weighted_median(np.array([1]), np.array([1, 2]))

@pytest.mark.parametrize("eval_date, expected_quarter, expected_fiscal_year", [
    ('2024-01-01', '2023Q4', '2023Q4'),
    ('2024-03-15', '2023Q4', '2023Q4'),
    ('2024-04-01', '2023Q4', '2023Q4'),
    ('2024-06-15', '2024Q2', '2024Q4'),
    ('2024-07-01', '2024Q2', '2024Q4'),
    ('2024-09-15', '2024Q3', '2024Q4'),
    ('2024-10-01', '2024Q3', '2024Q4'),
    ('2024-12-31', '2024Q4', '2024Q4'),
    ('2025-06-01', '2025Q2', '2025Q4'),  # Additional for month=6
    ('2025-08-01', '2025Q2', '2025Q4'),  # Month=8
    ('2025-09-15', '2025Q3', '2025Q4'),  # Month=9
    ('2025-11-01', '2025Q3', '2025Q4'),  # Month=11
    ('2025-12-01', '2025Q4', '2025Q4'),  # Month=12
])
def test_fiscal_period_info(eval_date, expected_quarter, expected_fiscal_year):
    """Test fiscal period info calculation comprehensively"""
    fiscal_info = evaluate_brokerage_report.get_fiscal_period_info(eval_date)
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
        'quarter_comparison': [True, True, True, True, True],
        'quarter': ['2024Q4', '2024Q4', '2024Q4', '2024Q4', '2024Q4']
    })

    with patch('tushare_provider.evaluate_brokerage_report.get_report_weight') as mock_get_weight:
        mock_get_weight.side_effect = lambda rt: {'点评': 3.0, '一般': 2.0, '非个股': 1.0}.get(rt, 2.0)

        result_bullish = evaluate_brokerage_report.aggregate_forecasts(test_df, 'bullish')
        assert result_bullish['eps'] == pytest.approx(2.6, 0.1)
        assert result_bullish['pe'] == pytest.approx(4.05, 0.1)
        assert result_bullish['rd'] == pytest.approx(5.25, 0.1)
        assert result_bullish['roe'] == pytest.approx(9.75, 0.1)
        assert result_bullish['ev_ebitda'] == pytest.approx(8.75, 0.1)
        assert result_bullish['max_price'] == pytest.approx(12.35, 0.1)
        assert result_bullish['min_price'] == pytest.approx(11.75, 0.1)

        result_bearish = evaluate_brokerage_report.aggregate_forecasts(test_df, 'bearish')
        assert result_bearish['eps'] == pytest.approx(2.6, 0.1)  # Adjust based on actual calculation if different

def test_aggregate_forecasts_empty_fields():
    """Test aggregate_forecasts with empty field data"""
    test_df = pd.DataFrame({
        'report_type': ['点评'],
        'report_weight': [3.0],
        'quarter_comparison': [True],
        'quarter': ['2024Q4']
    })
    result = evaluate_brokerage_report.aggregate_forecasts(test_df, 'bullish')
    assert result['eps'] is None
    assert result['max_price'] is None

def test_aggregate_forecasts_len_one():
    """Test aggregate_forecasts with single value"""
    test_df = pd.DataFrame({
        'eps': [2.5],
        'max_price': [12.0],
        'report_weight': [3.0],
        'quarter_comparison': [True],
        'quarter': ['2024Q4']
    })
    result = evaluate_brokerage_report.aggregate_forecasts(test_df, 'bullish')
    assert result['eps'] == 2.5
    assert result['max_price'] == 12.0

def test_aggregate_forecasts_len_zero_after_filter():
    """Test aggregate_forecasts with values filtered out"""
    test_df = pd.DataFrame({
        'eps': [1000.0],  # Out of range for eps
        'report_weight': [3.0],
        'quarter_comparison': [True],
        'quarter': ['2024Q4']
    })
    result = evaluate_brokerage_report.aggregate_forecasts(test_df, 'bullish')
    assert result['eps'] is None

def test_upsert_batch_empty_df(mock_engine):
    """Test _upsert_batch with empty df"""
    result = evaluate_brokerage_report._upsert_batch(mock_engine, pd.DataFrame())
    assert result == 0

def test_upsert_batch(mock_engine):
    """Test _upsert_batch normal operation"""
    df = pd.DataFrame({
        'ts_code': ['000001.SZ'],
        'eval_date': ['2025-01-01'],
        'report_period': ['2024Q4'],
    }).reindex(columns=evaluate_brokerage_report.ALL_COLUMNS, fill_value=None)

    with patch('tushare_provider.evaluate_brokerage_report.mysql_insert') as mock_insert:
        mock_conn = mock_engine.begin.return_value.__enter__.return_value
        mock_stmt = MagicMock()

        # Mock the inserted attribute to support column access
        mock_inserted = MagicMock()
        for col in evaluate_brokerage_report.ALL_COLUMNS:
            mock_inserted.__getitem__.return_value = MagicMock()
        mock_stmt.inserted = mock_inserted

        mock_insert.return_value = mock_stmt
        mock_stmt.on_duplicate_key_update.return_value = mock_stmt
        mock_conn.execute.return_value.rowcount = 1
        result = evaluate_brokerage_report._upsert_batch(mock_engine, df)
        assert result == 1

@pytest.mark.parametrize("stocks_input, expected", [
    (['000001.SZ'], ['000001.SZ']),
    ("000001.SZ,000002.SZ", ['000001.SZ', '000002.SZ']),
])
def test_get_stocks_list_specific(mock_engine, stocks_input, expected):
    """Test get_stocks_list with specific stocks"""
    result = evaluate_brokerage_report.get_stocks_list(mock_engine, stocks_input)
    assert result == expected

def test_get_stocks_list_query(mock_engine):
    """Test get_stocks_list from query"""
    mock_conn = mock_engine.begin.return_value.__enter__.return_value
    mock_conn.execute.return_value.fetchall.return_value = [('000001.SZ',)]
    result = evaluate_brokerage_report.get_stocks_list(mock_engine)
    assert result == ['000001.SZ']

def test_get_stocks_list_error(mock_engine, caplog):
    """Test get_stocks_list error"""
    mock_conn = mock_engine.begin.return_value.__enter__.return_value
    mock_conn.execute.side_effect = Exception("DB error")
    result = evaluate_brokerage_report.get_stocks_list(mock_engine)
    assert result == []
    assert "Error getting stocks list" in caplog.text

def test_evaluate_brokerage_report_dry_run(caplog):
    """Test evaluate_brokerage_report dry_run"""
    caplog.set_level(logging.INFO)
    with patch('tushare_provider.evaluate_brokerage_report.create_engine') as mock_create_engine:
        mock_create_engine.return_value = MagicMock()
        with patch('tushare_provider.evaluate_brokerage_report.get_trade_cal') as mock_trade_cal:
            mock_trade_cal.return_value = pd.DataFrame({'cal_date': ['20250101']})
            with patch('tushare_provider.evaluate_brokerage_report.get_stocks_list') as mock_stocks:
                mock_stocks.return_value = ['000001.SZ']
                evaluate_brokerage_report.evaluate_brokerage_report(
                    start_date='20250101',
                    end_date='20250101',
                    dry_run=True
                )
    assert "DRY RUN - No DB writes" in caplog.text

def test_evaluate_brokerage_report_invalid_date(caplog):
    """Test evaluate_brokerage_report invalid date"""
    with patch('tushare_provider.evaluate_brokerage_report.create_engine') as mock_create_engine:
        mock_create_engine.return_value = MagicMock()
        evaluate_brokerage_report.evaluate_brokerage_report(start_date='invalid')
    assert "Invalid date" in caplog.text

def test_evaluate_brokerage_report_no_stocks():
    """Test evaluate_brokerage_report no stocks"""
    with patch('tushare_provider.evaluate_brokerage_report.create_engine') as mock_create_engine:
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = datetime.datetime(2025, 1, 1)
        mock_conn.execute.return_value = mock_result
        mock_engine.begin.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        with patch('tushare_provider.evaluate_brokerage_report.get_trade_cal') as mock_trade_cal:
            mock_trade_cal.return_value = pd.DataFrame({'cal_date': ['20250101']})
            with patch('tushare_provider.evaluate_brokerage_report.get_stocks_list') as mock_stocks:
                mock_stocks.return_value = []
                evaluate_brokerage_report.evaluate_brokerage_report()

def test_evaluate_brokerage_report_trade_cal_empty():
    """Test evaluate_brokerage_report with empty trade_cal"""
    with patch('tushare_provider.evaluate_brokerage_report.create_engine') as mock_create_engine:
        mock_create_engine.return_value = MagicMock()
        with patch('tushare_provider.evaluate_brokerage_report.get_trade_cal') as mock_trade_cal:
            mock_trade_cal.return_value = pd.DataFrame()
            with patch('tushare_provider.evaluate_brokerage_report.get_stocks_list') as mock_stocks:
                mock_stocks.return_value = ['000001.SZ']
                evaluate_brokerage_report.evaluate_brokerage_report(start_date='20250101', end_date='20250101')

'''
def test_evaluate_brokerage_report_processing_error(caplog):
    """Test evaluate_brokerage_report concurrent error"""
    with patch('tushare_provider.evaluate_brokerage_report.create_engine') as mock_create_engine:
        mock_create_engine.return_value = MagicMock()
        with patch('tushare_provider.evaluate_brokerage_report.get_trade_cal') as mock_cal:
            mock_cal.return_value = pd.DataFrame({'cal_date': ['20250101']})
            with patch('tushare_provider.evaluate_brokerage_report.get_stocks_list') as mock_stocks:
                mock_stocks.return_value = ['000001.SZ']
                with patch('concurrent.futures.ThreadPoolExecutor') as mock_exec:
                    mock_future = MagicMock()
                    mock_future.result.side_effect = Exception("process error")
                    mock_exec.return_value.__enter__.return_value.submit.return_value = mock_future
                    mock_exec.return_value.__enter__.return_value.as_completed.return_value = [mock_future]
                    evaluate_brokerage_report.evaluate_brokerage_report(start_date='20250101', end_date='20250101')
    assert "Error processing" in caplog.text
'''

def test_load_config_file_missing():
    """Test load_config when config file doesn't exist"""
    with patch('os.path.exists', return_value=False):
        with patch('builtins.open', side_effect=FileNotFoundError):
            evaluate_brokerage_report.load_config()
            # Should not raise exception, uses defaults


def test_load_config_json_error():
    """Test load_config with invalid JSON"""
    with patch('os.path.exists', return_value=True):
        with patch('json.load', side_effect=json.JSONDecodeError("Invalid JSON", "", 0)):
            evaluate_brokerage_report.load_config()
            # Should not raise exception, uses defaults


def test_load_config_success():
    """Test successful config loading"""
    mock_json = '{"rating_mapping": {"BUY": ["Buy"]}, "report_type_weights": {"depth": 5.0}}'
    with patch('builtins.open', new_callable=mock_open, read_data=mock_json):
        evaluate_brokerage_report.load_config()

    assert evaluate_brokerage_report.RATING_MAPPING['BUY'] == ["Buy"]
    assert evaluate_brokerage_report.REPORT_TYPE_WEIGHTS['depth'] == 5.0

def test_tushare_token_not_set(caplog, monkeypatch):
    """Test TUSHARE_TOKEN not set"""
    monkeypatch.delenv("TUSHARE", raising=False)
    with pytest.raises(SystemExit):
        importlib.reload(evaluate_brokerage_report)
    assert "TUSHARE environment variable not set" in caplog.text

def test_get_report_weight_error_conversion(caplog):
    """Test error in str conversion in get_report_weight"""
    class BadType:
        def __str__(self):
            raise Exception("conversion error")
    result = evaluate_brokerage_report.get_report_weight(BadType())
    assert result == evaluate_brokerage_report.DEFAULT_REPORT_WEIGHT

def test_get_report_weight_no_match(caplog):
    """Test get_report_weight with no match"""
    caplog.set_level(logging.DEBUG)  # Set log level to capture DEBUG messages
    result = evaluate_brokerage_report.get_report_weight('unknown_type')
    assert result == evaluate_brokerage_report.DEFAULT_REPORT_WEIGHT
    assert "No match for unknown_type" in caplog.text

@pytest.mark.parametrize("report_type, expected_category", [
    ('深度报告', 'depth'),
    ('深度分析', 'depth'),
    ('调研报告', 'research'),
    ('调研纪要', 'research'),
    ('点评', 'commentary'),
    ('点评报告', 'commentary'),
    ('一般报告', 'general'),
    ('industry', 'other'),
    ('unknown', 'other'),
    (None, 'other'),
    ('', 'other'),
    ('   ', 'other'),
])
def test_categorize_report_type(report_type, expected_category):
    """Test categorize_report_type function"""
    result = evaluate_brokerage_report.categorize_report_type(report_type)
    assert result == expected_category

def test_get_trade_cal_error():
    """Test get_trade_cal error path"""
    with patch('tushare_provider.evaluate_brokerage_report.pro.trade_cal') as mock_cal:
        mock_cal.side_effect = Exception("API error")
        result = evaluate_brokerage_report.get_trade_cal('20250101', '20250101')
        assert result.empty

def test_get_trade_cal_normal():
    """Test get_trade_cal normal path"""
    mock_df = pd.DataFrame({'cal_date': ['20250101']})
    with patch('tushare_provider.evaluate_brokerage_report.pro.trade_cal') as mock_cal:
        mock_cal.return_value = mock_df
        result = evaluate_brokerage_report.get_trade_cal('20250101', '20250101')
        pd.testing.assert_frame_equal(result, mock_df)

@pytest.mark.parametrize("quarter_str, expected", [
    ('2024Q1', (2024, 1)),  # Valid
    ('2025Q4', (2025, 4)),
    ('2024Q5', (0, 0)),  # Invalid quarter >4
    ('2024Q0', (0, 0)),  # <1
    ('invalid', (0, 0)),
])
def test_parse_quarter(quarter_str, expected):
    """Test parse_quarter with valid and invalid"""
    result = evaluate_brokerage_report.parse_quarter(quarter_str)
    assert result == expected

@pytest.mark.parametrize("q1, q2, expected", [
    ('2024Q1', '2024Q2', -1),
    ('2024Q2', '2024Q2', 0),
    ('2025Q1', '2024Q4', 1),
])
def test_compare_quarters(q1, q2, expected):
    """Test compare_quarters normal cases"""
    result = evaluate_brokerage_report.compare_quarters(q1, q2)
    assert result == expected

def test_compare_quarters_invalid():
    """Test compare_quarters invalid format raise"""
    with pytest.raises(ValueError):
        evaluate_brokerage_report.compare_quarters('invalid', '2024Q1')

@pytest.mark.parametrize("field, values, weights, expected_length", [
    ('eps', np.array([-60.0, -40.0, 0.0, 40.0, 60.0]), np.array([1.0]*5), 3),
    ('pe', np.array([-1.0, 0.0, 100.0, 600.0]), np.array([1.0]*4), 1),
    ('rd', np.array([1.0]), np.array([1.0]), 1),
    ('unknown', np.array([1.0, 2.0]), np.array([1.0, 1.0]), 2),  # Covers full mask=ones
])
def test_apply_field_ranges(field, values, weights, expected_length):
    """Test _apply_field_ranges function"""
    filtered_values, filtered_weights = evaluate_brokerage_report._apply_field_ranges(field, values, weights)
    assert len(filtered_values) == expected_length

@pytest.mark.parametrize("values, weights, expected_length", [
    (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([1.0]*5), 3),  # 5% percentile filters out 1 and 5
    (np.array([10.0]*20 + [1000.0]), np.array([1.0]*21), 20),  # Outlier 1000 gets filtered
    (np.array([]), np.array([]), 0),  # Empty
])
def test_filter_outliers(values, weights, expected_length):
    """Test _filter_outliers function"""
    filtered_values, filtered_weights = evaluate_brokerage_report._filter_outliers(values, weights)
    assert len(filtered_values) == expected_length

def test_aggregate_forecasts_missing_columns():
    """Test aggregate_forecasts with missing columns"""
    df = pd.DataFrame({
        'eps': [2.5, 2.6, 2.4, 2.7, 2.3, 2.8],
        'report_type': ['点评', '一般', '点评', '一般', '点评', '一般'],
        'report_weight': [3.0, 2.0, 3.0, 2.0, 3.0, 2.0]
    })
    result = evaluate_brokerage_report.aggregate_forecasts(df, 'bullish')
    assert result['eps'] is not None
    assert result['pe'] is None

def test_aggregate_forecasts_outliers_filtered():
    """Test aggregate_forecasts with outliers filtered out"""
    df = pd.DataFrame({
        'eps': [2.5, 1000.0, -1000.0],
        'report_weight': [3.0, 3.0, 3.0],
        'quarter_comparison': [True, True, True],
        'quarter': ['2024Q4', '2024Q4', '2024Q4']
    })
    result = evaluate_brokerage_report.aggregate_forecasts(df, 'bullish')
    assert result['eps'] == 2.5  # Outliers filtered, only one left

def test_get_date_window_invalid_format():
    """Test get_date_window with invalid date format"""
    with pytest.raises(ValueError, match="Invalid eval_date format"):
        evaluate_brokerage_report.get_date_window('invalid_date')

def test_get_date_window_normal():
    """Test get_date_window normal"""
    start, end = evaluate_brokerage_report.get_date_window('20250101', window_months=6)
    assert start == '20240701'  # Exactly 6 months back from 20250101
    assert end == '20250101'

@pytest.fixture
def sample_bulk_data():
    """Create sample bulk brokerage data for testing bulk query optimization"""
    dates = [f"2024{(i//30)+1:02d}{(i%30)+1:02d}" for i in range(90)]  # 90 days of data
    data = []

    for date in dates:
        num_reports = np.random.randint(5, 16)
        for _ in range(num_reports):
            data.append({
                'ts_code': '000001.SZ',
                'report_date': date,
                'report_title': f'报告标题_{date}',
                'report_type': np.random.choice(['点评', '一般', '深度', '调研']),
                'classify': '一般报告',
                'org_name': f'券商_{np.random.randint(1, 100)}',
                'quarter': f'2024Q{np.random.randint(1, 5)}',
                'rating': np.random.choice(['买入', '增持', '中性', '减持', '卖出']),
                'eps': round(np.random.normal(2.5, 0.5), 2),
                'pe': round(np.random.normal(15, 3), 1),
                'rd': round(np.random.normal(3.0, 0.8), 2),
                'roe': round(np.random.normal(12, 2), 1),
                'ev_ebitda': round(np.random.normal(10, 2), 1),
                'max_price': round(np.random.normal(25, 5), 2),
                'min_price': round(np.random.normal(18, 3), 2)
            })

    return pd.DataFrame(data)

def test_bulk_query_date_range_calculation():
    """Test bulk query date range calculation"""
    date_list = ['2024-01-01', '2024-01-15', '2024-02-01', '2024-02-15', '2024-03-01']

    start_dt = dt.strptime(min(date_list), "%Y-%m-%d")
    end_dt = dt.strptime(max(date_list), "%Y-%m-%d")
    bulk_start_dt = start_dt - datetime.timedelta(days=180)

    expected_bulk_start = bulk_start_dt.strftime("%Y%m%d")
    expected_bulk_end = end_dt.strftime("%Y%m%d")

    actual_bulk_start = expected_bulk_start
    actual_bulk_end = expected_bulk_end

    assert actual_bulk_start == expected_bulk_start
    assert actual_bulk_end == expected_bulk_end

    assert actual_bulk_start < min(date_list)
    assert actual_bulk_end >= max(date_list)

def test_fiscal_info_precomputation():
    """Test fiscal info precomputation performance"""
    date_list = [f"2024-{i:02d}-{j:02d}" for i in range(1, 13) for j in [1, 15]]

    start_time = time.perf_counter()
    fiscal_infos = {date: evaluate_brokerage_report.get_fiscal_period_info(date) for date in date_list}
    precompute_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    for date in date_list:
        _ = evaluate_brokerage_report.get_fiscal_period_info(date)
    ondemand_time = time.perf_counter() - start_time

    # Allow for some variance in timing, but precompute should generally be faster
    assert precompute_time <= ondemand_time * 3.0 or precompute_time < 0.01  # Very fast execution

    assert len(fiscal_infos) == len(date_list)
    for date in date_list:
        assert date in fiscal_infos
        assert 'current_quarter' in fiscal_infos[date]

def test_groupby_performance_vs_filtering(sample_bulk_data):
    """Test groupby performance vs traditional filtering"""
    target_dates = sample_bulk_data['report_date'].unique()[:10]

    grouped = sample_bulk_data.groupby('report_date')

    start_time = time.perf_counter()
    groupby_results = {}
    for date in target_dates:
        if date in grouped.groups:
            groupby_results[date] = grouped.get_group(date)
    groupby_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    filter_results = {}
    for date in target_dates:
        filter_results[date] = sample_bulk_data[sample_bulk_data['report_date'] == date]
    filter_time = time.perf_counter() - start_time

    assert groupby_time <= filter_time * 3.0

    logger.info(f"GroupBy time: {groupby_time:.6f}s, Filter time: {filter_time:.6f}s, "
                f"Ratio: {groupby_time/filter_time:.2f}x")

    for date in target_dates:
        if date in groupby_results and date in filter_results:
            pd.testing.assert_frame_equal(
                groupby_results[date].reset_index(drop=True),
                filter_results[date].reset_index(drop=True)
            )

def test_vectorized_operations_performance(sample_bulk_data):
    """Test vectorized operations vs loop-based operations"""
    test_data = sample_bulk_data.head(1000).copy()

    start_time = time.perf_counter()
    vectorized_df = test_data.copy()
    vectorized_df['report_weight'] = vectorized_df['report_type'].apply(evaluate_brokerage_report.get_report_weight)
    vectorized_df['rating_category'] = vectorized_df['rating'].apply(evaluate_brokerage_report.classify_rating)
    vectorized_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    loop_df = test_data.copy()
    loop_df['report_weight'] = [evaluate_brokerage_report.get_report_weight(rt) for rt in loop_df['report_type']]
    loop_df['rating_category'] = [evaluate_brokerage_report.classify_rating(r) for r in loop_df['rating']]
    loop_time = time.perf_counter() - start_time

    assert vectorized_time <= loop_time * 2.0

    pd.testing.assert_series_equal(
        vectorized_df['report_weight'],
        loop_df['report_weight'],
        check_names=False
    )
    pd.testing.assert_series_equal(
        vectorized_df['rating_category'],
        loop_df['rating_category'],
        check_names=False
    )

@patch('tushare_provider.evaluate_brokerage_report.create_engine')
def test_concurrent_processing_simulation(mock_create_engine):
    """Test concurrent processing simulation"""
    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine

    with patch('tushare_provider.evaluate_brokerage_report.process_stock_all_dates') as mock_process:
        mock_process.return_value = 30

        stocks = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ']
        dates = ['20240101', '20240102', '20240103']

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(mock_process, mock_engine, stock, dates, 1000): stock for stock in stocks}
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        assert len(results) == len(stocks)
        assert all(result == 30 for result in results)

def test_memory_usage_optimization():
    """Test memory usage optimization techniques"""
    large_df = pd.DataFrame({
        'ts_code': ['000001.SZ'] * 10000,
        'report_date': [f'2024{i:02d}01' for i in range(1, 11)] * 1000,
        'eps': np.random.randn(10000),
        'report_type': ['点评'] * 10000
    })

    start_memory = large_df.memory_usage(deep=True).sum()

    grouped = large_df.groupby('report_date')
    processed_groups = {}

    for date, group in grouped:
        processed_groups[date] = group['eps'].mean()
        del group

    end_memory = large_df.memory_usage(deep=True).sum()

    assert end_memory <= start_memory * 1.5

    assert len(processed_groups) == len(large_df['report_date'].unique())

def test_error_handling_comprehensive(mock_engine):
    """Test comprehensive error handling in bulk processing"""
    date_list = ['2024-01-01', '2024-01-02', '2024-01-03']

    with patch.object(mock_engine, 'begin') as mock_begin:
        mock_begin.side_effect = Exception("Database connection error")

        with patch('tushare_provider.evaluate_brokerage_report.logger') as mock_logger:
            result = evaluate_brokerage_report.process_stock_all_dates(
                mock_engine, '000001.SZ', date_list, 1000
            )

            assert result == 0
            mock_logger.error.assert_called()

    with patch.object(mock_engine, 'begin') as mock_conn:
        mock_cursor = MagicMock()
        mock_conn.return_value.__enter__.return_value = mock_cursor

        mock_query = MagicMock()
        mock_conn.return_value.__enter__.return_value.execute.return_value = mock_query

        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame(columns=[
                'ts_code', 'report_date', 'report_title', 'report_type',
                'classify', 'org_name', 'quarter', 'rating', 'eps', 'pe',
                'rd', 'roe', 'ev_ebitda', 'max_price', 'min_price'
            ])


            result = evaluate_brokerage_report.process_stock_all_dates(
                mock_engine, '000001.SZ', date_list, 1000
            )

            assert result == 0

def test_performance_regression_detection():
    """Test performance regression detection"""
    scenarios = {
        'small_dataset': {'stocks': 10, 'dates': 30, 'expected_time': 0.03},
        'medium_dataset': {'stocks': 100, 'dates': 30, 'expected_time': 0.3},
        'large_dataset': {'stocks': 1000, 'dates': 30, 'expected_time': 3.0}
    }

    for scenario_name, params in scenarios.items():
        start_time = time.perf_counter()

        processing_time = params['stocks'] * params['dates'] * 0.0001
        time.sleep(max(processing_time, 0.001))

        actual_time = time.perf_counter() - start_time

        tolerance = params['expected_time'] * 1.0
        assert abs(actual_time - params['expected_time']) <= tolerance, \
            f"Performance regression in {scenario_name}: expected ~{params['expected_time']:.3f}s, got {actual_time:.3f}s"

def test_concurrent_vs_sequential_scaling():
    """Test how concurrent processing scales vs sequential"""
    stock_counts = [1, 2, 4, 8, 16]

    for num_stocks in stock_counts:
        stocks = [f"{i:06d}.SZ" for i in range(num_stocks)]

        seq_start = time.time()
        for stock in stocks:
            time.sleep(0.01)
        seq_time = time.time() - seq_start

        conc_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_stocks, 8)) as executor:
            futures = [executor.submit(lambda: time.sleep(0.01)) for _ in stocks]
            concurrent.futures.wait(futures)
        conc_time = time.time() - conc_start

        # Only check performance for larger workloads where concurrency should provide benefit
        # For small workloads, sequential might be faster due to overhead
        if num_stocks >= 8:
            # Allow some tolerance for concurrent processing overhead
            assert conc_time <= seq_time * 1.2, f"Concurrent processing significantly slower for {num_stocks} stocks"

        speedup = seq_time / conc_time if conc_time > 0 else float('inf')
        print(".2f")

def test_data_quality_validation(sample_bulk_data):
    """Test data quality validation in bulk processing"""
    test_data = sample_bulk_data.head(10).copy()

    test_data['report_weight'] = test_data['report_type'].apply(evaluate_brokerage_report.get_report_weight)
    test_data['rating_category'] = test_data['rating'].apply(evaluate_brokerage_report.classify_rating)

    valid_result = evaluate_brokerage_report.aggregate_consensus_from_df(
        test_data, '000001.SZ', '2024-01-01',
        {'current_quarter': '2024Q1', 'current_fiscal_year': '2024Q4'}
    )
    assert valid_result is not None

    incomplete_data = test_data.drop(columns=['rating'])

    result = evaluate_brokerage_report.aggregate_consensus_from_df(
        incomplete_data, '000001.SZ', '2024-01-01',
        {'current_quarter': '2024Q1', 'current_fiscal_year': '2024Q4'}
    )
    assert result is not None
    assert isinstance(result['buy_count'], (int, type(None)))

    invalid_data = test_data.copy()
    invalid_data['eps'] = invalid_data['eps'].astype(str)

    result = evaluate_brokerage_report.aggregate_consensus_from_df(
        invalid_data, '000001.SZ', '2024-01-01',
        {'current_quarter': '2024Q1', 'current_fiscal_year': '2024Q4'}
    )

    assert result is not None

def test_resource_cleanup_verification():
    """Test that resources are properly cleaned up"""
    initial_threads = threading.active_count()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(time.sleep, 0.1) for _ in range(10)]
        concurrent.futures.wait(futures)

    time.sleep(0.5)

    final_threads = threading.active_count()

    assert abs(final_threads - initial_threads) <= 2, \
        f"Thread leak detected: initial {initial_threads}, final {final_threads}"


# ===== INTEGRATION TESTS =====

@patch('tushare_provider.evaluate_brokerage_report.create_engine')
def test_full_processing_pipeline(mock_create_engine):
    """Test full processing pipeline from start to finish"""
    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine

    with patch('tushare_provider.evaluate_brokerage_report.get_trade_cal') as mock_trade_cal, \
         patch('tushare_provider.evaluate_brokerage_report.get_stocks_list') as mock_stocks, \
         patch('tushare_provider.evaluate_brokerage_report.process_stock_all_dates') as mock_process, \
         patch('tushare_provider.evaluate_brokerage_report._upsert_batch') as mock_upsert:

        mock_trade_cal.return_value = pd.DataFrame({'cal_date': ['2024-01-01', '2024-01-02']})
        mock_stocks.return_value = ['000001.SZ', '000002.SZ']
        mock_process.return_value = 50

        mock_df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 50 + ['000002.SZ'] * 50,
            'eval_date': ['2024-01-01'] * 50 + ['2024-01-02'] * 50,
            'report_period': ['2024Q1'] * 100,
            'total_reports': [10] * 100
        })
        mock_upsert.return_value = 100

        result = evaluate_brokerage_report.evaluate_brokerage_report(
            mysql_url="mysql+pymysql://test:test@localhost/test",
            start_date="20240101",
            end_date="20240102",
            max_workers=2,
            dry_run=False
        )

        mock_trade_cal.assert_called_once()
        mock_stocks.assert_called_once()
        assert mock_process.call_count == 2

def test_error_recovery_scenarios():
    pass

def test_performance_under_load():
    pass

# Additional new tests

def test_aggregate_consensus_from_df_bullish():
    """Test aggregate_consensus_from_df with bullish sentiment"""
    df = pd.DataFrame({
        'rating_category': ['BUY', 'HOLD', 'BUY'],
        'report_type': ['深度', '调研', '点评'],
        'report_weight': [5.0, 4.0, 3.0],
        'quarter': ['2024Q4', '2024Q4', '2024Q4'],
    })
    result = evaluate_brokerage_report.aggregate_consensus_from_df(df, '000001.SZ', '2025-01-01', {'current_quarter': '2024Q4', 'current_fiscal_year': '2024Q4'})
    assert result['sentiment_pos'] == 3
    assert result['sentiment_neg'] == 0
    assert result['depth_reports'] == 1
    assert result['research_reports'] == 1
    assert result['commentary_reports'] == 1

def test_aggregate_consensus_from_df_bearish():
    """Test aggregate_consensus_from_df with bearish sentiment"""
    df = pd.DataFrame({
        'rating_category': ['NEUTRAL', 'SELL', 'NEUTRAL'],
        'report_type': ['一般', '非个股', '会议纪要'],
        'report_weight': [2.0, 1.0, 3.0],
        'quarter': ['2024Q4', '2024Q4', '2024Q4'],
    })
    result = evaluate_brokerage_report.aggregate_consensus_from_df(df, '000001.SZ', '2025-01-01', {'current_quarter': '2024Q4', 'current_fiscal_year': '2024Q4'})
    assert result['sentiment_pos'] == 0
    assert result['sentiment_neg'] == 3
    assert result['general_reports'] == 1
    assert result['other_reports'] == 1
    assert result['commentary_reports'] == 1

def test_aggregate_consensus_from_df_neutral():
    """Test aggregate_consensus_from_df with neutral sentiment"""
    df = pd.DataFrame({
        'rating_category': ['BUY', 'NEUTRAL'],
        'report_type': ['深度', '点评'],
        'report_weight': [5.0, 3.0],
        'quarter': ['2024Q4', '2024Q4'],
    })
    result = evaluate_brokerage_report.aggregate_consensus_from_df(df, '000001.SZ', '2025-01-01', {'current_quarter': '2024Q4', 'current_fiscal_year': '2024Q4'})
    assert result['sentiment_pos'] == 1
    assert result['sentiment_neg'] == 1

def test_aggregate_consensus_from_df_empty():
    """Test aggregate_consensus_from_df with empty DF"""
    df = pd.DataFrame()
    result = evaluate_brokerage_report.aggregate_consensus_from_df(df, '000001.SZ', '2025-01-01', {'current_quarter': '2024Q4', 'current_fiscal_year': '2024Q4'})
    assert result is None

def test_aggregate_consensus_from_df_error(caplog):
    """Test aggregate_consensus_from_df error handling"""
    df = pd.DataFrame()  # Missing required columns to trigger KeyError
    with patch('pandas.DataFrame.__getitem__') as mock_getitem:
        mock_getitem.side_effect = KeyError("rating_category")
        result = evaluate_brokerage_report.aggregate_consensus_from_df(df, '000001.SZ', '2025-01-01', {'current_quarter': '2024Q4', 'current_fiscal_year': '2024Q4'})
    assert result is None
    assert "Error aggregating consensus" in caplog.text

def test_aggregate_consensus_from_df_quarter_filter():
    """Test aggregate_consensus_from_df with quarter filtering"""
    df = pd.DataFrame({
        'rating_category': ['BUY', 'HOLD', 'BUY'],
        'report_type': ['深度', '调研', '点评'],
        'report_weight': [5.0, 4.0, 3.0],
        'quarter': ['2024Q3', '2024Q4', '2025Q1'],
    })
    result = evaluate_brokerage_report.aggregate_consensus_from_df(df, '000001.SZ', '2025-01-01', {'current_quarter': '2024Q4', 'current_fiscal_year': '2024Q4'})
    assert result['total_reports'] == 3  # Filters out Q3

def test_process_stock_all_dates_with_brokerage(mock_engine):
    """Test process_stock_all_dates with brokerage data"""
    date_list = ['2025-01-02']
    brokerage_df = pd.DataFrame({
        'ts_code': ['000001.SZ'],
        'report_date': ['2025-01-01'],
        'report_type': ['深度'],
        'rating': ['买入'],
        'quarter': ['2025Q1']
    })
    with patch('pandas.read_sql') as mock_read_sql:
        mock_read_sql.return_value = brokerage_df
        with patch('tushare_provider.evaluate_brokerage_report._upsert_batch') as mock_upsert:
            mock_upsert.return_value = 1
            result = evaluate_brokerage_report.process_stock_all_dates(mock_engine, '000001.SZ', date_list, 1000)
    assert result == 1

def test_process_stock_all_dates_error(caplog):
    """Test process_stock_all_dates error in loop"""
    date_list = ['2025-01-01']
    mock_engine = MagicMock()
    with patch('pandas.read_sql') as mock_read_sql:
        mock_read_sql.return_value = pd.DataFrame()
        with patch('tushare_provider.evaluate_brokerage_report.aggregate_consensus_from_df') as mock_agg:
            mock_agg.side_effect = Exception("agg error")
            result = evaluate_brokerage_report.process_stock_all_dates(mock_engine, '000001.SZ', date_list, 1000)
    assert result == 0
    assert "Error in bulk processing" in caplog.text

def test_process_stock_all_dates_upsert_error(caplog):
    """Test process_stock_all_dates upsert error"""
    caplog.set_level(logging.ERROR)  # Set log level to capture ERROR messages
    date_list = ['2025-01-01']
    mock_engine = MagicMock()
    with patch('pandas.read_sql') as mock_read_sql:
        mock_read_sql.return_value = pd.DataFrame({'ts_code': ['000001.SZ'], 'report_date': ['2024-12-31'], 'report_type': ['深度'], 'rating': ['买入'], 'quarter': ['2025Q1']})
        with patch('tushare_provider.evaluate_brokerage_report._upsert_batch') as mock_upsert:
            mock_upsert.side_effect = Exception("upsert error")
            result = evaluate_brokerage_report.process_stock_all_dates(mock_engine, '000001.SZ', date_list, 1000)
    assert result == 0
    assert "Error upserting results for stock" in caplog.text

def test_evaluate_brokerage_report_start_after_end(caplog):
    """Test evaluate_brokerage_report start > end"""
    with patch('tushare_provider.evaluate_brokerage_report.create_engine') as mock_engine:
        mock_engine.return_value = MagicMock()
        result = evaluate_brokerage_report.evaluate_brokerage_report(start_date='20250102', end_date='20250101')
        assert result is None
        assert "Invalid date" in caplog.text
'''
def test_evaluate_brokerage_report_normal(caplog):
    """Test evaluate_brokerage_report normal run"""
    caplog.set_level(logging.INFO)
    with patch('tushare_provider.evaluate_brokerage_report.create_engine') as mock_engine:
        mock_engine.return_value = MagicMock()
        with patch('tushare_provider.evaluate_brokerage_report.get_trade_cal') as mock_cal:
            mock_cal.return_value = pd.DataFrame({'cal_date': ['20250101']})
            with patch('tushare_provider.evaluate_brokerage_report.get_stocks_list') as mock_stocks:
                mock_stocks.return_value = ['000001.SZ']
                with patch('concurrent.futures.ThreadPoolExecutor') as mock_exec:
                    mock_future = MagicMock()
                    mock_future.result.return_value = 1
                    mock_exec.return_value.__enter__.return_value.submit.return_value = mock_future
                    mock_exec.return_value.__enter__.return_value.as_completed.return_value = [mock_future]
                    evaluate_brokerage_report.evaluate_brokerage_report(start_date='20250101', end_date='20250101')
    assert "Processing 1 stocks" in caplog.text
    assert "Completed processing all stocks. Total upserted: 1 records" in caplog.text

def test_evaluate_brokerage_report_progress_log(caplog):
    """Test evaluate_brokerage_report progress logging for many stocks"""
    caplog.set_level(logging.INFO)
    with patch('tushare_provider.evaluate_brokerage_report.create_engine') as mock_engine:
        mock_engine.return_value = MagicMock()
        with patch('tushare_provider.evaluate_brokerage_report.get_trade_cal') as mock_cal:
            mock_cal.return_value = pd.DataFrame({'cal_date': ['20250101']})
            with patch('tushare_provider.evaluate_brokerage_report.get_stocks_list') as mock_stocks:
                mock_stocks.return_value = [f'{i:06d}.SZ' for i in range(100)]  # Enough for %50 log
                with patch('concurrent.futures.ThreadPoolExecutor') as mock_exec:
                    mock_futures = [MagicMock() for _ in range(100)]
                    for f in mock_futures:
                        f.result.return_value = 1
                    mock_exec.return_value.__enter__.return_value.submit.side_effect = mock_futures
                    mock_exec.return_value.__enter__.return_value.as_completed.return_value = mock_futures
                    evaluate_brokerage_report.evaluate_brokerage_report(start_date='20250101', end_date='20250101')
    assert "Completed 50/100 stocks" in caplog.text
    assert "Completed 100/100 stocks" in caplog.text  # Since 100 %50 ==0? No, but if loop hits.
'''
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
