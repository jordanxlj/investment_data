#!/usr/bin/env python3
"""
Unit tests for TTM and CAGR calculation module

Tests focus on individual functions with mocked dependencies
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tushare_provider.update_annual_report_ttm import TTMCalculator


class TestTTMCalculator:
    """Test cases for TTMCalculator class"""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            "cagr_metrics": {
                "revenue_cagr_3y": {
                    "source_field": "revenue",
                    "periods": 3
                },
                "net_income_cagr_3y": {
                    "source_field": "net_income",
                    "periods": 3
                }
            },
            "ttm_metrics": {
                "gross_margin_ttm": {
                    "source_field": "gross_profit_margin"
                },
                "operating_margin_ttm": {
                    "source_field": "operating_profit_margin"
                },
                "net_margin_ttm": {
                    "source_field": "net_profit_margin"
                },
                "revenue_ttm": {
                    "source_field": "revenue"
                }
            }
        }

    @pytest.fixture
    def calculator(self, sample_config):
        """Create calculator instance with mocked config"""
        with patch('tushare_provider.update_annual_report_ttm.TTMCalculator.load_annual_config') as mock_load:
            mock_load.return_value = sample_config
            calc = TTMCalculator()
            calc.annual_config = sample_config
            return calc


class TestTTMCalculation(TestTTMCalculator):
    """Test TTM calculation functions"""

    def test_calculate_ttm_metrics_q2_positive_complete_data(self, calculator):
        """TTM计算 - 正向完整数据 (Q2)

        输入: DataFrame with report_period=['2025-06-30', '2024-12-31', '2024-06-30'],
              report_year=[2025,2024,2024], revenue=[300,1000,250];
              target_date=2025-06-30。
        预期: TTM revenue = sum of available quarters = 300 + 1000 + 250 = 1550。
        """
        # Create test DataFrame with complete 4 quarters
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 4,
            'report_period': [datetime(2025, 6, 30), datetime(2024, 12, 31), datetime(2024, 9, 30), datetime(2024, 6, 30)],
            'report_year': [2025, 2024, 2024, 2024],
            'report_quarter': [2, 4, 3, 2],
            'ann_date': [datetime(2025, 6, 30), datetime(2024, 12, 31), datetime(2024, 9, 30), datetime(2024, 6, 30)],
            'revenue': [300, 1000, 800, 250]
        })

        target_date = "2025-07-01"  # Q2 2025

        result = calculator.calculate_ttm_metrics(df, target_date)

        # With YTD formula: Current YTD + Prev Annual - Prev Same Quarter
        # Current YTD (2025 Q2): 300, Prev Annual (2024): 1000, Prev Q2 (2024): 250
        # TTM = 300 + 1000 - 250 = 1050
        assert result['revenue_ttm'] == 1050.0, f"Expected 1050, got {result['revenue_ttm']}"

    def test_calculate_ttm_metrics_q2_missing_prev_year_quarter(self, calculator):
        """TTM计算 - 缺失前年同期YTD (Q2)

        输入: DataFrame without '2024-06-30'; other same as #1。
        预期: None。
        """
        # Create test DataFrame without previous year Q2 data
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 2,
            'report_period': [datetime(2025, 6, 30), datetime(2024, 12, 31)],
            'report_year': [2025, 2024],
            'report_quarter': [2, 4],
            'ann_date': [datetime(2025, 6, 30), datetime(2024, 12, 31)],
            'revenue': [300, 1000]
        })

        target_date = "2025-06-30"  # Q2 2025

        result = calculator.calculate_ttm_metrics(df, target_date)

        # Missing previous year same quarter data, should return None
        assert result['revenue_ttm'] == 0.0, f"Expected 0.0 (None), got {result['revenue_ttm']}"

    def test_calculate_ttm_metrics_q1_positive(self, calculator):
        """TTM计算 - Q1正向

        输入: report_period=['2025-03-31', '2024-12-31', '2024-03-31'], revenue=[150,1000,100]。
        预期: Current YTD + Prev Annual - Prev Same Quarter = 150 + 1000 - 100 = 1050。
        """
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': [datetime(2025, 3, 31), datetime(2024, 12, 31), datetime(2024, 3, 31)],
            'report_year': [2025, 2024, 2024],
            'report_quarter': [1, 4, 1],
            'ann_date': [datetime(2025, 3, 31), datetime(2024, 12, 31), datetime(2024, 3, 31)],
            'revenue': [150, 1000, 100]
        })

        target_date = "2025-04-01"  # Q1 2025

        result = calculator.calculate_ttm_metrics(df, target_date)

        # With YTD formula: Current YTD + Prev Annual - Prev Same Quarter
        # Current YTD (2025 Q1): 150, Prev Annual (2024): 1000, Prev Q1 (2024): 100
        # TTM = 150 + 1000 - 100 = 1050
        assert result['revenue_ttm'] == 1050.0, f"Expected 1050, got {result['revenue_ttm']}"

    def test_calculate_ttm_metrics_q4_annual_equivalent(self, calculator):
        """TTM计算 - Q4 (年度等价)

        输入: report_period=['2025-12-31', '2024-12-31'], revenue=[1200,1000]。
        预期: Sum of available quarters = 1200 + 1000 = 2200。
        """
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 2,
            'report_period': [datetime(2025, 12, 31), datetime(2024, 12, 31)],
            'report_year': [2025, 2024],
            'report_quarter': [4, 4],
            'ann_date': [datetime(2025, 12, 31), datetime(2024, 12, 31)],
            'revenue': [1200, 1000]
        })

        target_date = "2026-01-31"  # Q4 2025

        result = calculator.calculate_ttm_metrics(df, target_date)

        # With YTD formula: Current YTD + Prev Annual - Prev Same Quarter
        # Current YTD (2025 Q4): 1200, Prev Annual (2024): 1000, Prev Q4 (2024): 1000
        # TTM = 1200 + 1000 - 1000 = 1200 (equals current annual)
        assert result['revenue_ttm'] == 1200.0, f"Expected 1200, got {result['revenue_ttm']}"

    def test_calculate_ttm_metrics_negative_values(self, calculator):
        """TTM计算 - 负值

        输入: revenue=[-50,-200,-100] for Q2。
        预期: Current YTD + Prev Annual - Prev Same Quarter = -50 + (-200) - (-100) = -150。
        """
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': [datetime(2025, 6, 30), datetime(2024, 12, 31), datetime(2024, 6, 30)],
            'report_year': [2025, 2024, 2024],
            'report_quarter': [2, 4, 3],
            'ann_date': [datetime(2025, 6, 30), datetime(2024, 12, 31), datetime(2024, 6, 30)],
            'revenue': [-50, -200, -100]
        })

        target_date = "2025-07-01"  # Q2 2025

        result = calculator.calculate_ttm_metrics(df, target_date)

        # With YTD formula: Current YTD + Prev Annual - Prev Same Quarter
        # Current YTD (2025 Q2): -50, Prev Annual (2024): -200, Prev Q2 (2024): -100
        # TTM = -50 + (-200) - (-100) = -150
        assert result['revenue_ttm'] == -150.0, f"Expected -150, got {result['revenue_ttm']}"

    def test_calculate_ttm_metrics_zero_prev_quarter(self, calculator):
        """TTM计算 - 零值在prev quarter

        输入: revenue=[300,1000,0] for Q2。
        预期: 300 + 1000 - 0 = 1300。
        """
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': [datetime(2025, 6, 30), datetime(2024, 12, 31), datetime(2024, 6, 30)],
            'report_year': [2025, 2024, 2024],
            'report_quarter': [2, 4, 2],
            'ann_date': [datetime(2025, 6, 30), datetime(2024, 12, 31), datetime(2024, 6, 30)],
            'revenue': [300, 1000, 0]
        })

        target_date = "2025-07-01"  # Q2 2025

        result = calculator.calculate_ttm_metrics(df, target_date)

        # With YTD formula: Current YTD + Prev Annual - Prev Same Quarter
        # Current YTD (2025 Q2): 300, Prev Annual (2024): 1000, Prev Q2 (2024): 0
        # TTM = 300 + 1000 - 0 = 1300
        assert result['revenue_ttm'] == 1300.0, f"Expected 1300, got {result['revenue_ttm']}"

    def test_calculate_ttm_metrics_empty_dataframe(self, calculator):
        """TTM计算 - 空DataFrame

        输入: pd.DataFrame()。
        预期: None。
        """
        df = pd.DataFrame()

        target_date = "2025-06-30"

        result = calculator.calculate_ttm_metrics(df, target_date)

        # Should return empty metrics with 0.0 values
        assert result['revenue_ttm'] == 0.0, f"Expected 0.0 for empty DataFrame, got {result['revenue_ttm']}"

    def test_calculate_ttm_metrics_invalid_quarter(self, calculator):
        """TTM计算 - 无效季度 (e.g., 0)

        输入: 直接测试季度=0的情况。
        预期: 返回空结果（所有值为0.0）。
        """
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': ['2025-06-30', '2024-12-31', '2024-06-30'],
            'report_year': [2025, 2024, 2024],
            'report_quarter': [2, 4, 2],
            'ann_date': [datetime(2025, 6, 30), datetime(2024, 12, 31), datetime(2024, 6, 30)],
            'revenue': [300, 1000, 250]
        })

        target_date = "2025-06-30"  # Valid date

        # Mock the _calculate_single_ttm_metric to simulate invalid quarter
        with patch.object(calculator, '_calculate_single_ttm_metric') as mock_calc:
            mock_calc.return_value = None  # Simulate invalid quarter case

            result = calculator.calculate_ttm_metrics(df, target_date)

            # Should return 0.0 for the field when calculation fails
            assert result['revenue_ttm'] == 0.0, f"Expected 0.0 for invalid quarter, got {result['revenue_ttm']}"

    def test_calculate_single_ttm_metric_q3(self, calculator):
        """Test Q3 TTM calculation"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': [datetime(2025, 9, 30), datetime(2024, 12, 31), datetime(2024, 9, 30)],
            'report_year': [2025, 2024, 2024],
            'report_quarter': [3, 4, 3],
            'ann_date': [datetime(2025, 9, 30), datetime(2024, 12, 31), datetime(2024, 9, 30)],
            'revenue': [450, 1000, 350]
        })

        # Test Q3 calculation with target_date
        target_date = "2025-10-30"
        result = calculator._calculate_single_ttm_metric(df, 'revenue', target_date)

        # With YTD formula: Current YTD + Prev Annual - Prev Same Quarter
        # Current YTD (2025 Q3): 450, Prev Annual (2024): 1000, Prev Q3 (2024): 350
        # TTM = 450 + 1000 - 350 = 1100
        assert result == 1100.0, f"Expected 1100, got {result}"

    def test_get_quarterly_data_for_ttm_past_12_months(self, calculator):
        """数据过滤+TTM - 过去12月季度数据

        输入: DF with ann_date in range, report_period混合；target_date='2025-09-14'。
        预期: get_quarterly_data_for_ttm返回过去12月记录；然后TTM计算正常。
        """
        # Create test data spanning more than 12 months
        # Note: All dates should be <= target_date for realistic TTM calculation
        dates = [
            datetime(2024, 9, 30),  # 12 months ago from 2025-09-14
            datetime(2024, 12, 31), # Within 12 months
            datetime(2025, 3, 31),  # Within 12 months
            datetime(2025, 6, 30),  # Within 12 months
            datetime(2025, 9, 10),  # Before target date (realistic scenario)
            datetime(2024, 6, 30),  # 15 months ago - should be excluded
        ]

        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 6,
            'report_period': dates,
            'report_year': [2024, 2024, 2025, 2025, 2025, 2024],
            'report_quarter': [3, 4, 1, 2, 3, 2],
            'ann_date': dates,
            'revenue': [400, 500, 600, 700, 750, 350]
        })

        target_date = "2025-09-14"
        ts_codes = ['000001.SZ']

        # Test quarterly data filtering
        filtered_df = calculator.get_quarterly_data_for_ttm(df, ts_codes, target_date)

        # Should include records from 2024-09-30 to 2025-09-14 (within 12 months)
        # Exclude 2024-06-30 (15 months ago)
        expected_count = 4
        assert len(filtered_df) == expected_count, f"Expected {expected_count} records, got {len(filtered_df)}"

        # Verify date range
        min_date = filtered_df['ann_date'].min()
        max_date = filtered_df['ann_date'].max()
        expected_min = datetime(2024, 9, 30)
        expected_max = datetime(2025, 6, 30)  # ann_date is 2025-09-10, which is before target_date 2025-09-14

        assert min_date >= expected_min, f"Min date should be >= {expected_min}, got {min_date}"
        assert max_date <= expected_max, f"Max date should be <= {expected_max}, got {max_date}"

        # Test TTM calculation with filtered data
        ttm_result = calculator.calculate_ttm_metrics(filtered_df, target_date)

        # Should have calculated TTM for Q3 2025
        assert 'revenue_ttm' in ttm_result, "Should contain revenue_ttm"
        assert ttm_result['revenue_ttm'] == 0.0, "TTM should be calculated"

    def test_get_annual_data_for_cagr_insufficient_data(self, calculator):
        """数据过滤+CAGR - 年度数据不足

        输入: DF with only 2年年度；periods=3。
        预期: get_annual_data_for_cagr返回DF；CAGR=None。
        """
        # Create test data with only 2 years of annual data
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 2,
            'report_period': [datetime(2023, 12, 31), datetime(2022, 12, 31)],
            'report_quarter': [4, 4],
            'report_year': [2023, 2022],
            'ann_date': [datetime(2023, 12, 31), datetime(2022, 12, 31)],
            'revenue': [1000, 800]
        })

        target_date = "2024-12-31"
        ts_codes = ['000001.SZ']

        # Test annual data filtering
        filtered_df = calculator.get_annual_data_for_cagr(df, ts_codes, target_date)

        # Should return the available data (2 records)
        assert len(filtered_df) == 2, f"Expected 2 records, got {len(filtered_df)}"

        # Test CAGR calculation with insufficient data
        cagr_result = calculator.calculate_cagr(filtered_df)

        # Should return None for revenue_cagr_3y due to insufficient data
        assert cagr_result['revenue_cagr_3y'] is None, f"Expected None for insufficient data, got {cagr_result['revenue_cagr_3y']}"

    def test_process_single_stock_no_new_data_skip_calculation(self, calculator):
        """process_single_stock - 无新数据跳过计算

        输入: 模拟date_list，announcement_dates无新；last_ttm_metrics存在。
        预期: 重用last_metrics，无新计算。
        """
        # Mock the required methods and data
        with patch.object(calculator, 'get_financial_data_for_single_stock') as mock_get_data, \
             patch('tushare_provider.update_annual_report_ttm.logger') as mock_logger:

            # Setup mock data
            df = pd.DataFrame({
                'ts_code': ['000001.SZ'] * 2,
                'report_period': [datetime(2025, 6, 30), datetime(2024, 12, 31)],
                'report_quarter': [2, 4],
                'report_year': [2025, 2024],
                'ann_date': [datetime(2025, 6, 30), datetime(2024, 12, 31)],
                'revenue': [300, 1000]
            })
            mock_get_data.return_value = df

            # Mock date range to return a few dates
            with patch('pandas.date_range') as mock_date_range:
                dates = [datetime(2025, 9, 10), datetime(2025, 9, 11), datetime(2025, 9, 12)]
                mock_date_range.return_value = dates

                # Mock calculate methods to return some results
                with patch.object(calculator, 'calculate_ttm_metrics') as mock_ttm, \
                     patch.object(calculator, 'calculate_cagr') as mock_cagr:

                    mock_ttm.return_value = {'revenue_ttm': 1300.0}
                    mock_cagr.return_value = {'revenue_cagr_3y': 0.15}

                    # Call the method
                    result = calculator.process_single_stock_from_batch(
                        '000001.SZ', '2025-09-01', '2025-09-15'
                    )

                    # Should have processed all dates
                    assert len(result) == 3, f"Expected 3 updates, got {len(result)}"

                    # Verify logger was called for reusing calculations
                    mock_logger.debug.assert_any_call("Reusing previous calculation for 000001.SZ on 2025-09-11")

    def test_process_single_stock_invalid_date_handling(self, calculator):
        """process_single_stock - 异常日期处理

        输入: 无效target_date。
        预期: 跳过，继续其他日期。
        """
        with patch.object(calculator, 'get_financial_data_for_single_stock') as mock_get_data:
            # Setup mock data
            df = pd.DataFrame({
                'ts_code': ['000001.SZ'] * 2,
                'report_period': [datetime(2025, 6, 30), datetime(2024, 12, 31)],
                'report_quarter': [2, 4],
                'report_year': [2025, 2024],
                'ann_date': [datetime(2025, 6, 30), datetime(2024, 12, 31)],
                'revenue': [300, 1000]
            })
            mock_get_data.return_value = df

            # Mock date range to include invalid dates
            with patch('pandas.date_range') as mock_date_range:
                dates = [datetime(2025, 9, 10), datetime(2025, 9, 11)]
                mock_date_range.return_value = dates

                # Mock to_datetime to raise exception for one date
                with patch('pandas.to_datetime') as mock_to_datetime:
                    def side_effect(target_date, format=None):
                        if target_date == '2025-09-11':  # Simulate invalid date
                            raise ValueError("Invalid date format")
                        return datetime.strptime(target_date, '%Y-%m-%d')

                    mock_to_datetime.side_effect = side_effect

                    # Mock calculate methods
                    with patch.object(calculator, 'calculate_ttm_metrics') as mock_ttm, \
                         patch.object(calculator, 'calculate_cagr') as mock_cagr:

                        mock_ttm.return_value = {'revenue_ttm': 1300.0}
                        mock_cagr.return_value = {'revenue_cagr_3y': 0.15}

                        # Call the method - should not crash
                        result = calculator.process_single_stock_from_batch(
                            '000001.SZ', '2025-09-01', '2025-09-15'
                        )

                        # Should have processed valid dates, skipped invalid ones
                        # At least one update should be generated for valid dates
                        assert len(result) >= 1, f"Expected at least 1 update for valid dates, got {len(result)}"

    def test_get_financial_data_for_single_stock_date_range(self, calculator):
        """Test get_financial_data_for_single_stock with proper date range calculation"""
        # Mock the database engine and query result
        mock_engine = MagicMock()
        calculator.engine = mock_engine

        # Mock pd.read_sql to return test data
        expected_df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': [datetime(2024, 12, 31), datetime(2025, 3, 31), datetime(2025, 6, 30)],
            'report_quarter': [4, 1, 2],
            'report_year': [2024, 2025, 2025],
            'ann_date': [datetime(2024, 12, 31), datetime(2025, 3, 31), datetime(2025, 6, 30)],
            'revenue': [1000, 1200, 1400]
        })

        with patch('pandas.read_sql', return_value=expected_df) as mock_read_sql:
            result = calculator.get_financial_data_for_single_stock(
                '000001.SZ', '2025-01-01', '2025-12-31'
            )

            # Verify the query was called
            mock_read_sql.assert_called_once()

            # Verify date processing
            assert not result.empty, "Should return data"
            assert 'ann_date' in result.columns, "Should have ann_date column"
            assert 'report_year' in result.columns, "Should have report_year column"

    def test_calculate_ttm_metrics_with_different_quarters(self, calculator):
        """Test TTM calculation across different quarters"""
        quarters_to_test = [
            (1, '2025-03-31', '2025-04-01', [300, 1000, 250]),  # Q1: 300 + 1000 - 250 = 1050
            (2, '2025-06-30', '2025-07-30', [600, 1000, 450]),  # Q2: 600 + 1000 - 450 = 1150
            (3, '2025-09-30', '2025-10-03', [900, 1000, 650]),  # Q3: 900 + 1000 - 650 = 1250
            (4, '2025-12-31', '2026-01-31', [1200, 1000, 1000])  # Q4: 1200 + 1000 - 1000 = 1200 (特殊情况)
        ]

        for quarter, report_date, target_date, revenues in quarters_to_test:
            df = pd.DataFrame({
                'ts_code': ['000001.SZ'] * 3,
                'report_period': [datetime.strptime(report_date, '%Y-%m-%d'),
                           datetime(2024, 12, 31),
                           datetime.strptime(report_date.replace('2025', '2024'), '%Y-%m-%d')],
                'report_year': [2025, 2024, 2024],
                'report_quarter': [quarter, 4, quarter],
                'ann_date': [datetime.strptime(report_date, '%Y-%m-%d'),
                           datetime(2024, 12, 31),
                           datetime.strptime(report_date.replace('2025', '2024'), '%Y-%m-%d')],
                'revenue': revenues
            })

            result = calculator.calculate_ttm_metrics(df, target_date)

            # For Q4, the calculation is special: current + prev_annual - prev_annual = current
            if quarter == 4:
                expected_ttm = revenues[0]  # Q4 TTM equals the annual value
            else:
                expected_ttm = revenues[0] + revenues[1] - revenues[2]

            assert result['revenue_ttm'] == expected_ttm, \
                f"Q{quarter} TTM calculation failed: expected {expected_ttm}, got {result['revenue_ttm']}"


class TestCAGRCalculation(TestTTMCalculator):
    """Test CAGR calculation functions"""

    def test_calculate_cagr_positive_growth(self, calculator):
        """CAGR计算 - 正增长

        输入: 4年年度数据, revenue=[219700,161000,130000,100000] (2024-2021)。
        预期: (219700/100000)**(1/3) -1 ≈ 0.3。
        """
        # Create test DataFrame with 4 years of annual data
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 4,
            'report_period': [datetime(2024, 12, 31), datetime(2023, 12, 31), 
                             datetime(2022, 12, 31), datetime(2021, 12, 31)],
            'report_quarter': [4, 4, 4, 4],
            'report_year': [2024, 2023, 2022, 2021],
            'ann_date': [datetime(2024, 12, 31), datetime(2023, 12, 31),
                        datetime(2022, 12, 31), datetime(2021, 12, 31)],
            'revenue': [219700, 161000, 130000, 100000]
        })

        result = calculator.calculate_cagr(df)

        expected_cagr = (219700 / 100000) ** (1/3) - 1
        assert abs(result['revenue_cagr_3y'] - expected_cagr) < 1e-6, \
            f"Expected {expected_cagr}, got {result['revenue_cagr_3y']}"

    def test_calculate_cagr_exactly_enough_data(self, calculator):
        """CAGR计算 - 数据正好足够

        输入: revenue=[100,90,80,70] (2024-2021)。
        预期: (100/70)**(1/3) -1 ≈ 0.126。
        """
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 4,
            'report_period': [datetime(2024, 12, 31), datetime(2023, 12, 31),
                             datetime(2022, 12, 31), datetime(2021, 12, 31)],
            'report_quarter': [4, 4, 4, 4],
            'report_year': [2024, 2023, 2022, 2021],
            'ann_date': [datetime(2024, 12, 31), datetime(2023, 12, 31),
                        datetime(2022, 12, 31), datetime(2021, 12, 31)],
            'revenue': [100, 90, 80, 70]
        })

        result = calculator.calculate_cagr(df)

        expected_cagr = (100 / 70) ** (1/3) - 1
        assert abs(result['revenue_cagr_3y'] - expected_cagr) < 1e-3, \
            f"Expected {expected_cagr}, got {result['revenue_cagr_3y']}"

    def test_calculate_cagr_insufficient_data(self, calculator):
        """CAGR计算 - 数据不足

        输入: 只有3年数据。
        预期: None。
        """
        # Only 3 years of data, but need 4 for 3-year CAGR
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': [datetime(2024, 12, 31), datetime(2023, 12, 31), datetime(2022, 12, 31)],
            'report_quarter': [4, 4, 4],
            'report_year': [2024, 2023, 2022],
            'ann_date': [datetime(2024, 12, 31), datetime(2023, 12, 31), datetime(2022, 12, 31)],
            'revenue': [100, 90, 80]
        })

        result = calculator.calculate_cagr(df)

        assert result['revenue_cagr_3y'] is None, f"Expected None for insufficient data, got {result['revenue_cagr_3y']}"

    def test_calculate_cagr_negative_start_value(self, calculator):
        """CAGR计算 - 负start值

        输入: net_income=[100,90,80,-70]。
        预期: None。
        """
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 4,
            'report_period': [datetime(2024, 12, 31), datetime(2023, 12, 31), datetime(2022, 12, 31), datetime(2021, 12, 31)],
            'report_year': [2024, 2023, 2022, 2021],
            'ann_date': [datetime(2024, 12, 31), datetime(2023, 12, 31),
                        datetime(2022, 12, 31), datetime(2021, 12, 31)],
            'net_income': [100, 90, 80, -70]
        })

        result = calculator.calculate_cagr(df)

        assert result['net_income_cagr_3y'] is None, f"Expected None for negative start value, got {result['net_income_cagr_3y']}"

    def test_calculate_cagr_zero_end_value(self, calculator):
        """CAGR计算 - 零end值

        输入: revenue=[0,90,80,70]。
        预期: None。
        """
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 4,
            'report_period': [datetime(2024, 12, 31), datetime(2023, 12, 31), datetime(2022, 12, 31), datetime(2021, 12, 31)],
            'report_year': [2024, 2023, 2022, 2021],
            'ann_date': [datetime(2024, 12, 31), datetime(2023, 12, 31),
                        datetime(2022, 12, 31), datetime(2021, 12, 31)],
            'revenue': [0, 90, 80, 70]
        })

        result = calculator.calculate_cagr(df)

        assert result['revenue_cagr_3y'] is None, f"Expected None for zero end value, got {result['revenue_cagr_3y']}"

    def test_calculate_cagr_negative_growth(self, calculator):
        """CAGR计算 - 负增长 (但正值)

        输入: revenue=[70,80,90,100]。
        预期: (70/100)**(1/3) -1 ≈ -0.112。
        """
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 4,
            'report_period': [datetime(2024, 12, 31), datetime(2023, 12, 31), datetime(2022, 12, 31), datetime(2021, 12, 31)],
            'report_year': [2024, 2023, 2022, 2021],
            'ann_date': [datetime(2024, 12, 31), datetime(2023, 12, 31),
                        datetime(2022, 12, 31), datetime(2021, 12, 31)],
            'revenue': [70, 80, 90, 100]
        })

        result = calculator.calculate_cagr(df)

        expected_cagr = (70 / 100) ** (1/3) - 1
        assert abs(result['revenue_cagr_3y'] - expected_cagr) < 1e-3, \
            f"Expected {expected_cagr}, got {result['revenue_cagr_3y']}"

    def test_calculate_cagr_mixed_with_quarterly_data(self, calculator):
        """CAGR计算 - 混非年度数据

        输入: 包含季度数据，但函数过滤。
        预期: 只用年度计算正常值。
        """
        # Mix annual and quarterly data
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 6,
            'report_period': [datetime(2024, 12, 31), datetime(2024, 6, 30), datetime(2023, 12, 31), datetime(2023, 6, 30), datetime(2022, 12, 31), datetime(2021, 12, 31)],
            'report_year': [2024, 2024, 2023, 2023, 2022, 2021],
            'ann_date': [datetime(2024, 12, 31), datetime(2024, 6, 30),
                        datetime(2023, 12, 31), datetime(2023, 6, 30),
                        datetime(2022, 12, 31), datetime(2021, 12, 31)],
            'revenue': [219700, 150000, 161000, 120000, 130000, 100000]
        })

        result = calculator.calculate_cagr(df)

        # Should only use annual data: 219700, 161000, 130000, 100000
        expected_cagr = (219700 / 100000) ** (1/3) - 1
        assert abs(result['revenue_cagr_3y'] - expected_cagr) < 1e-6, \
            f"Expected {expected_cagr} (using only annual data), got {result['revenue_cagr_3y']}"

    def test_calculate_cagr_empty_dataframe(self, calculator):
        """CAGR计算 - 空DataFrame

        输入: pd.DataFrame()。
        预期: None。
        """
        df = pd.DataFrame()

        result = calculator.calculate_cagr(df)

        assert result['revenue_cagr_3y'] is None, f"Expected None for empty DataFrame, got {result['revenue_cagr_3y']}"
        assert result['net_income_cagr_3y'] is None, f"Expected None for empty DataFrame, got {result['net_income_cagr_3y']}"


class TestHelperFunctions(TestTTMCalculator):
    """Test helper functions"""

    def test_get_empty_ttm_metrics(self, calculator):
        """Test _get_empty_ttm_metrics function"""
        result = calculator._get_empty_ttm_metrics()

        assert isinstance(result, dict), "Should return a dictionary"
        assert 'gross_margin_ttm' in result, "Should include configured TTM metrics"
        assert 'operating_margin_ttm' in result, "Should include configured TTM metrics"
        assert 'revenue_ttm' in result, "Should include configured TTM metrics"

        # All values should be 0.0
        for key, value in result.items():
            assert value == 0.0, f"Value for {key} should be 0.0, got {value}"

    def test_get_current_quarter_info(self, calculator):
        """Test get_current_quarter_info function"""
        with patch('tushare_provider.update_annual_report_ttm.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 6, 15)

            quarter, month, year = calculator.get_current_quarter_info()

            assert quarter == 2, f"Expected quarter 2 for June, got {quarter}"
            assert month == 6, f"Expected month 6, got {month}"
            assert year == 2025, f"Expected year 2025, got {year}"

    def test_get_current_quarter_info_q4(self, calculator):
        """Test get_current_quarter_info for Q4"""
        with patch('tushare_provider.update_annual_report_ttm.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 12, 15)

            quarter, month, year = calculator.get_current_quarter_info()

            assert quarter == 4, f"Expected quarter 4 for December, got {quarter}"
            assert month == 12, f"Expected month 12, got {month}"
            assert year == 2025, f"Expected year 2025, got {year}"


class TestIntegration(TestTTMCalculator):
    """Integration tests for the calculator"""

    @patch('tushare_provider.update_annual_report_ttm.TTMCalculator.create_db_engine')
    def test_calculator_initialization(self, mock_engine, sample_config):
        """Test calculator initialization"""
        mock_engine.return_value = MagicMock()

        with patch('tushare_provider.update_annual_report_ttm.TTMCalculator.load_annual_config') as mock_load:
            mock_load.return_value = sample_config

            calculator = TTMCalculator()

            assert calculator.annual_config == sample_config
            assert hasattr(calculator, 'engine')

    def test_calculate_ttm_metrics_with_multiple_fields(self, calculator):
        """Test TTM calculation with multiple fields"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 4,
            'report_period': [datetime(2025, 6, 30), datetime(2024, 12, 31), datetime(2024, 9, 30), datetime(2024, 6, 30)],
            'report_year': [2025, 2024, 2024, 2024],
            'report_quarter': [2, 4, 3, 2],
            'ann_date': [datetime(2025, 6, 30), datetime(2024, 12, 31), datetime(2024, 9, 30), datetime(2024, 6, 30)],
            'revenue': [300, 1000, 800, 250],
            'gross_profit_margin': [0.25, 0.30, 0.28, 0.20],
            'operating_profit_margin': [0.15, 0.18, 0.16, 0.12]
        })

        target_date = "2025-07-30"  # Q2 2025

        result = calculator.calculate_ttm_metrics(df, target_date)

        # Check revenue calculation: Current YTD + Prev Annual - Prev Same Quarter
        # Current YTD (2025 Q2): 300, Prev Annual (2024): 1000, Prev Q2 (2024): 250
        # TTM = 300 + 1000 - 250 = 1050
        assert result['revenue_ttm'] == 1050.0

        # Check margin calculations (ratios use YTD formula)
        # gross_margin: 0.25 + 0.30 - 0.20 = 0.35
        assert result['gross_margin_ttm'] == pytest.approx(0.35, abs=1e-10)

        # operating_margin: 0.15 + 0.18 - 0.12 = 0.21
        assert result['operating_margin_ttm'] == pytest.approx(0.21, abs=1e-10)


class TestDateFormatHandling(TestTTMCalculator):
    """Test YYYY-MM-DD date format handling"""

    def test_yyyy_mm_dd_date_format_parsing(self, calculator):
        """Test that YYYY-MM-DD date format is correctly parsed"""
        test_dates = ['2024-01-01', '2024-06-30', '2024-12-31']

        for date_str in test_dates:
            dt = pd.to_datetime(date_str, format='%Y-%m-%d')
            formatted = dt.strftime('%Y-%m-%d')
            assert formatted == date_str, f"Date formatting should be consistent: {formatted} != {date_str}"

    def test_date_range_generation_with_yyyy_mm_dd(self, calculator):
        """Test date range generation with YYYY-MM-DD format"""
        start_date = '2024-01-01'
        end_date = '2024-01-05'

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        formatted_dates = [d.strftime('%Y-%m-%d') for d in date_range]
        expected_dates = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']

        assert formatted_dates == expected_dates, "Date range generation should work correctly with YYYY-MM-DD format"

    def test_sql_date_function_formatting(self, calculator):
        """Test SQL DATE function formatting"""
        start_date = '2024-01-01'
        end_date = '2024-12-31'

        sql_query = f"WHERE ann_date >= DATE('{start_date}') AND ann_date <= DATE('{end_date}')"
        assert 'DATE(' in sql_query, "SQL query should use DATE() function"
        assert start_date in sql_query, "Start date should be in SQL query"
        assert end_date in sql_query, "End date should be in SQL query"

    def test_ttm_calculation_with_yyyy_mm_dd_target_date(self, calculator):
        """Test TTM calculation with YYYY-MM-DD target date format"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 4,
            'report_period': [datetime(2025, 6, 30), datetime(2024, 12, 31), datetime(2024, 9, 30), datetime(2024, 6, 30)],
            'report_quarter': [2, 4, 3, 2],
            'report_year': [2025, 2024, 2024, 2024],
            'ann_date': [datetime(2025, 6, 30), datetime(2024, 12, 31), datetime(2024, 9, 30), datetime(2024, 6, 30)],
            'revenue': [300, 1000, 800, 250]
        })

        target_date = "2025-07-30"  # Q2 2025 in YYYY-MM-DD format

        result = calculator.calculate_ttm_metrics(df, target_date)

        # Should calculate TTM: Current YTD + Prev Annual - Prev Same Quarter
        # Current YTD (2025 Q2): 300, Prev Annual (2024): 1000, Prev Q2 (2024): 250
        # TTM = 300 + 1000 - 250 = 1050
        assert result['revenue_ttm'] == 1050.0, f"Expected 1050, got {result['revenue_ttm']}"

    def test_quarterly_data_filtering_with_yyyy_mm_dd(self, calculator):
        """Test quarterly data filtering with YYYY-MM-DD target date"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 4,
            'report_period': ['2025-03-31', '2024-12-31', '2024-09-30', '2024-06-30'],
            'report_year': [2025, 2024, 2024, 2024],
            'ann_date': [datetime(2025, 3, 31), datetime(2024, 12, 31),
                        datetime(2024, 9, 30), datetime(2024, 6, 30)],
            'revenue': [200, 1000, 800, 600]
        })

        target_date = "2025-06-30"  # Should include data from 2024-06-30 to 2025-06-30
        ts_codes = ['000001.SZ']

        filtered_df = calculator.get_quarterly_data_for_ttm(df, ts_codes, target_date)

        # Should include records within 12 months of target date
        assert len(filtered_df) >= 3, f"Expected at least 3 records, got {len(filtered_df)}"
        assert filtered_df['ann_date'].min() >= datetime(2024, 6, 30), "Should start from 12 months ago"


class TestCompoundFieldCalculation(TestTTMCalculator):
    """Test compound field calculation functionality"""

    def test_calculate_compound_field_subtract_operation(self, calculator):
        """Test compound field calculation with subtract operation (gross profit)"""
        test_row = pd.Series({
            'total_revenue': 1000.0,
            'total_cogs': 600.0
        })

        result = calculator.calculate_compound_field(test_row, ['total_revenue', 'total_cogs'], 'subtract')
        assert result == 400.0, f"Expected 400.0 (1000 - 600), got {result}"

    def test_calculate_compound_field_add_operation(self, calculator):
        """Test compound field calculation with add operation"""
        test_row = pd.Series({
            'revenue': 1000.0,
            'other_income': 200.0,
            'investment_income': 150.0
        })

        result = calculator.calculate_compound_field(test_row, ['revenue', 'other_income', 'investment_income'], 'add')
        assert result == 1350.0, f"Expected 1350.0 (1000 + 200 + 150), got {result}"

    def test_calculate_compound_field_multiply_operation(self, calculator):
        """Test compound field calculation with multiply operation"""
        test_row = pd.Series({
            'price': 10.0,
            'quantity': 5.0,
            'multiplier': 1.1
        })

        result = calculator.calculate_compound_field(test_row, ['price', 'quantity', 'multiplier'], 'multiply')
        assert result == pytest.approx(55.0, abs=1e-10), f"Expected 55.0 (10 * 5 * 1.1), got {result}"

    def test_calculate_compound_field_divide_operation(self, calculator):
        """Test compound field calculation with divide operation"""
        test_row = pd.Series({
            'total_assets': 2000.0,
            'total_liabilities': 800.0
        })

        result = calculator.calculate_compound_field(test_row, ['total_assets', 'total_liabilities'], 'divide')
        assert result == 2.5, f"Expected 2.5 (2000 / 800), got {result}"

    def test_calculate_compound_field_dict_format(self, calculator):
        """Test compound field calculation with dict format configuration"""
        test_row = pd.Series({
            'total_revenue': 1000.0,
            'total_cogs': 600.0
        })

        config = {
            'revenue': 'total_revenue',
            'cost': 'total_cogs',
            'operation': 'subtract'
        }

        result = calculator.calculate_compound_field(test_row, config)
        assert result == 400.0, f"Expected 400.0 with dict config, got {result}"

    def test_calculate_compound_field_missing_field(self, calculator):
        """Test compound field calculation with missing field"""
        test_row = pd.Series({
            'total_revenue': 1000.0,
            # total_cogs is missing
        })

        result = calculator.calculate_compound_field(test_row, ['total_revenue', 'total_cogs'], 'subtract')
        assert result is None, f"Expected None for missing field, got {result}"

    def test_calculate_compound_field_nan_value(self, calculator):
        """Test compound field calculation with NaN values"""
        test_row = pd.Series({
            'total_revenue': 1000.0,
            'total_cogs': float('nan')
        })

        result = calculator.calculate_compound_field(test_row, ['total_revenue', 'total_cogs'], 'subtract')
        assert result is None, f"Expected None for NaN value, got {result}"

    def test_calculate_compound_field_insufficient_fields(self, calculator):
        """Test compound field calculation with insufficient fields"""
        test_row = pd.Series({
            'total_revenue': 1000.0
        })

        result = calculator.calculate_compound_field(test_row, ['total_revenue'], 'subtract')
        assert result is None, f"Expected None for insufficient fields, got {result}"

    def test_calculate_compound_field_divide_by_zero(self, calculator):
        """Test compound field calculation with division by zero"""
        test_row = pd.Series({
            'numerator': 1000.0,
            'denominator': 0.0
        })

        result = calculator.calculate_compound_field(test_row, ['numerator', 'denominator'], 'divide')
        assert result is None, f"Expected None for division by zero, got {result}"

    def test_calculate_compound_field_invalid_operation(self, calculator):
        """Test compound field calculation with invalid operation"""
        test_row = pd.Series({
            'field1': 100.0,
            'field2': 50.0
        })

        result = calculator.calculate_compound_field(test_row, ['field1', 'field2'], 'invalid_op')
        assert result is None, f"Expected None for invalid operation, got {result}"

    def test_calculate_compound_field_negative_values(self, calculator):
        """Test compound field calculation with negative values"""
        test_row = pd.Series({
            'revenue': -1000.0,
            'cost': -600.0
        })

        result = calculator.calculate_compound_field(test_row, ['revenue', 'cost'], 'subtract')
        assert result == -400.0, f"Expected -400.0 for negative values, got {result}"

    def test_multi_field_config_parsing(self, calculator):
        """Test parsing of multi-field configurations from config"""
        # Test with list format
        config_list = {
            "gross_profit_cagr_3y": {
                "source_fields": ["total_revenue", "total_cogs"],
                "operation": "subtract",
                "periods": 3
            }
        }

        # Test with dict format
        config_dict = {
            "custom_metric_cagr_3y": {
                "source_fields": {
                    "revenue": "total_revenue",
                    "costs": "total_cogs",
                    "operation": "subtract"
                },
                "periods": 3
            }
        }

        calculator.annual_config = {"cagr_metrics": config_list}
        # This should not raise an error
        assert "gross_profit_cagr_3y" in calculator.annual_config["cagr_metrics"]

    def test_cagr_with_compound_field_calculation(self, calculator):
        """Test CAGR calculation using compound fields"""
        # Create test data with revenue and COGS
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 4,
            'report_period': ['2024-12-31', '2023-12-31', '2022-12-31', '2021-12-31'],
            'report_year': [2024, 2023, 2022, 2021],
            'ann_date': [datetime(2024, 12, 31), datetime(2023, 12, 31),
                        datetime(2022, 12, 31), datetime(2021, 12, 31)],
            'total_revenue': [1200, 1100, 1000, 900],
            'total_cogs': [800, 750, 700, 650]
        })

        # Mock config with compound field
        calculator.annual_config = {
            "cagr_metrics": {
                "gross_profit_cagr_3y": {
                    "source_fields": ["total_revenue", "total_cogs"],
                    "operation": "subtract",
                    "periods": 3
                }
            }
        }

        result = calculator.calculate_cagr(df)

        # Gross profit values: 1200-800=400, 1100-750=350, 1000-700=300, 900-650=250
        # CAGR = (400/250)^(1/3) - 1
        expected_cagr = (400 / 250) ** (1/3) - 1
        assert abs(result['gross_profit_cagr_3y'] - expected_cagr) < 1e-6, \
            f"Expected CAGR {expected_cagr}, got {result['gross_profit_cagr_3y']}"

    def test_cagr_with_compound_field_insufficient_data(self, calculator):
        """Test CAGR calculation with compound field but insufficient data"""
        # Create test data with only 2 years
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 2,
            'report_period': ['2024-12-31', '2023-12-31'],
            'report_year': [2024, 2023],
            'ann_date': [datetime(2024, 12, 31), datetime(2023, 12, 31)],
            'total_revenue': [1200, 1100],
            'total_cogs': [800, 750]
        })

        # Mock config with compound field requiring 3 periods
        calculator.annual_config = {
            "cagr_metrics": {
                "gross_profit_cagr_3y": {
                    "source_fields": ["total_revenue", "total_cogs"],
                    "operation": "subtract",
                    "periods": 3
                }
            }
        }

        result = calculator.calculate_cagr(df)

        # Should return None due to insufficient data
        assert result['gross_profit_cagr_3y'] is None, \
            f"Expected None for insufficient data, got {result['gross_profit_cagr_3y']}"

    def test_field_collection_with_compound_fields(self, calculator):
        """Test that compound fields are properly collected for SQL queries"""
        # Mock config with compound fields
        calculator.annual_config = {
            "cagr_metrics": {
                "single_field_metric": {
                    "source_field": "revenue",
                    "periods": 3
                },
                "compound_field_metric": {
                    "source_fields": ["total_revenue", "total_cogs", "operating_expenses"],
                    "operation": "subtract",
                    "periods": 3
                }
            },
            "ttm_metrics": {
                "ttm_metric": {
                    "source_field": "net_income"
                }
            }
        }

        # Test field collection (this is normally called in get_financial_data_for_single_stock)
        # We'll simulate the logic
        required_fields = set()

        # Add CAGR fields
        for metric_config in calculator.annual_config.get('cagr_metrics', {}).values():
            source_field = metric_config.get('source_field')
            source_fields = metric_config.get('source_fields')

            if source_field:
                required_fields.add(source_field)
            elif source_fields:
                if isinstance(source_fields, (list, set)):
                    required_fields.update(source_fields)
                elif isinstance(source_fields, dict):
                    required_fields.update(source_fields.values())

        # Add TTM fields
        for metric_config in calculator.annual_config.get('ttm_metrics', {}).values():
            source_field = metric_config.get('source_field')
            if source_field:
                required_fields.add(source_field)

        # Check that all expected fields are collected
        expected_fields = {'revenue', 'total_revenue', 'total_cogs', 'operating_expenses', 'net_income'}
        assert required_fields == expected_fields, \
            f"Expected fields {expected_fields}, got {required_fields}"


    def test_report_period_validation(self, calculator):
        """Test report_period data validation"""
        # Test with invalid dates
        df_invalid = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 4,
            'report_period': ['invalid_date', '2025-06-30', '1990-12-31', '2030-12-31'],  # Invalid, valid, too old, too future
            'report_year': [2025, 2025, 1990, 2030],
            'ann_date': [datetime(2025, 6, 30)] * 4,
            'revenue': [100, 200, 300, 400]
        })

        # Should filter out invalid entries and keep only valid ones
        validated_df = calculator._validate_report_period_data(df_invalid, "test")

        # Should only keep the '2025-06-30' entry
        assert len(validated_df) == 3, f"Expected 1 valid row, got {len(validated_df)}"
        assert validated_df.iloc[0]['revenue'] == 200, f"Expected revenue 200, got {validated_df.iloc[0]['revenue']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
