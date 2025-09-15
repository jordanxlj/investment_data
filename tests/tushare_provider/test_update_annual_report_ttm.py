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
              current_year=2025, current_quarter=2。
        预期: TTM revenue = 300 + 1000 - 250 = 1050。
        """
        # Create test DataFrame
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': ['2025-06-30', '2024-12-31', '2024-06-30'],
            'report_year': [2025, 2024, 2024],
            'ann_date': [datetime(2025, 6, 30), datetime(2024, 12, 31), datetime(2024, 6, 30)],
            'revenue': [300, 1000, 250]
        })

        target_date = "20250630"  # Q2 2025

        result = calculator.calculate_ttm_metrics(df, target_date)

        assert result['revenue_ttm'] == 1050.0, f"Expected 1050, got {result['revenue_ttm']}"

    def test_calculate_ttm_metrics_q2_missing_prev_year_quarter(self, calculator):
        """TTM计算 - 缺失前年同期YTD (Q2)

        输入: DataFrame without '2024-06-30'; other same as #1。
        预期: None。
        """
        # Create test DataFrame without previous year Q2 data
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 2,
            'report_period': ['2025-06-30', '2024-12-31'],
            'report_year': [2025, 2024],
            'ann_date': [datetime(2025, 6, 30), datetime(2024, 12, 31)],
            'revenue': [300, 1000]
        })

        target_date = "20250630"  # Q2 2025

        result = calculator.calculate_ttm_metrics(df, target_date)

        assert result['revenue_ttm'] == 0.0, f"Expected None (0.0), got {result['revenue_ttm']}"

    def test_calculate_ttm_metrics_q1_positive(self, calculator):
        """TTM计算 - Q1正向

        输入: report_period=['2025-03-31', '2024-12-31', '2024-03-31'], revenue=[150,1000,100]。
        预期: 150 + 1000 - 100 = 1050。
        """
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': ['2025-03-31', '2024-12-31', '2024-03-31'],
            'report_year': [2025, 2024, 2024],
            'ann_date': [datetime(2025, 3, 31), datetime(2024, 12, 31), datetime(2024, 3, 31)],
            'revenue': [150, 1000, 100]
        })

        target_date = "20250331"  # Q1 2025

        result = calculator.calculate_ttm_metrics(df, target_date)

        assert result['revenue_ttm'] == 1050.0, f"Expected 1050, got {result['revenue_ttm']}"

    def test_calculate_ttm_metrics_q4_annual_equivalent(self, calculator):
        """TTM计算 - Q4 (年度等价)

        输入: report_period=['2025-12-31', '2024-12-31'], revenue=[1200,1000] (prev quarter is annual)。
        预期: 1200 + 1000 - 1000 = 1200。
        """
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 2,
            'report_period': ['2025-12-31', '2024-12-31'],
            'report_year': [2025, 2024],
            'ann_date': [datetime(2025, 12, 31), datetime(2024, 12, 31)],
            'revenue': [1200, 1000]
        })

        target_date = "20251231"  # Q4 2025

        result = calculator.calculate_ttm_metrics(df, target_date)

        assert result['revenue_ttm'] == 1200.0, f"Expected 1200, got {result['revenue_ttm']}"

    def test_calculate_ttm_metrics_negative_values(self, calculator):
        """TTM计算 - 负值

        输入: revenue=[-50,-200,-100] for Q2。
        预期: -50 + (-200) - (-100) = -150。
        """
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': ['2025-06-30', '2024-12-31', '2024-06-30'],
            'report_year': [2025, 2024, 2024],
            'ann_date': [datetime(2025, 6, 30), datetime(2024, 12, 31), datetime(2024, 6, 30)],
            'revenue': [-50, -200, -100]
        })

        target_date = "20250630"  # Q2 2025

        result = calculator.calculate_ttm_metrics(df, target_date)

        assert result['revenue_ttm'] == -150.0, f"Expected -150, got {result['revenue_ttm']}"

    def test_calculate_ttm_metrics_zero_prev_quarter(self, calculator):
        """TTM计算 - 零值在prev quarter

        输入: revenue=[300,1000,0] for Q2。
        预期: 300 + 1000 - 0 = 1300。
        """
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': ['2025-06-30', '2024-12-31', '2024-06-30'],
            'report_year': [2025, 2024, 2024],
            'ann_date': [datetime(2025, 6, 30), datetime(2024, 12, 31), datetime(2024, 6, 30)],
            'revenue': [300, 1000, 0]
        })

        target_date = "20250630"  # Q2 2025

        result = calculator.calculate_ttm_metrics(df, target_date)

        assert result['revenue_ttm'] == 1300.0, f"Expected 1300, got {result['revenue_ttm']}"

    def test_calculate_ttm_metrics_empty_dataframe(self, calculator):
        """TTM计算 - 空DataFrame

        输入: pd.DataFrame()。
        预期: None。
        """
        df = pd.DataFrame()

        target_date = "20250630"

        result = calculator.calculate_ttm_metrics(df, target_date)

        # Should return empty metrics with 0.0 values
        assert result['revenue_ttm'] == 0.0, f"Expected 0.0 for empty DataFrame, got {result['revenue_ttm']}"

    def test_calculate_ttm_metrics_invalid_quarter(self, calculator):
        """TTM计算 - 无效季度 (e.g., 0)

        输入: current_quarter=0; valid DF。
        预期: None。
        """
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': ['2025-06-30', '2024-12-31', '2024-06-30'],
            'report_year': [2025, 2024, 2024],
            'ann_date': [datetime(2025, 6, 30), datetime(2024, 12, 31), datetime(2024, 6, 30)],
            'revenue': [300, 1000, 250]
        })

        # Use a date that would result in quarter 0 (invalid)
        target_date = "20250000"  # Invalid date

        result = calculator.calculate_ttm_metrics(df, target_date)

        assert result['revenue_ttm'] == 0.0, f"Expected 0.0 for invalid quarter, got {result['revenue_ttm']}"

    def test_calculate_single_ttm_metric_q3(self, calculator):
        """Test Q3 TTM calculation"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': ['2025-09-30', '2024-12-31', '2024-09-30'],
            'report_year': [2025, 2024, 2024],
            'ann_date': [datetime(2025, 9, 30), datetime(2024, 12, 31), datetime(2024, 9, 30)],
            'revenue': [450, 1000, 350]
        })

        # Test Q3 calculation: 450 + 1000 - 350 = 1100
        result = calculator._calculate_single_ttm_metric(df, 'revenue', 2025, 3)

        assert result == 1100.0, f"Expected 1100 for Q3, got {result}"


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
            'report_period': ['2024-12-31', '2023-12-31', '2022-12-31', '2021-12-31'],
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
            'report_period': ['2024-12-31', '2023-12-31', '2022-12-31', '2021-12-31'],
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
            'report_period': ['2024-12-31', '2023-12-31', '2022-12-31'],
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
            'report_period': ['2024-12-31', '2023-12-31', '2022-12-31', '2021-12-31'],
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
            'report_period': ['2024-12-31', '2023-12-31', '2022-12-31', '2021-12-31'],
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
            'report_period': ['2024-12-31', '2023-12-31', '2022-12-31', '2021-12-31'],
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
            'report_period': ['2024-12-31', '2024-06-30', '2023-12-31', '2023-06-30', '2022-12-31', '2021-12-31'],
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

    def test_get_weight_matrix_q1(self, calculator):
        """Test get_weight_matrix for Q1"""
        with patch('tushare_provider.update_annual_report_ttm.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 1, 15)

            result = calculator.get_weight_matrix(1, 1)

            # Should use previous year annual only for Q1
            assert 'annual_2024' in result, "Should include previous year annual for Q1"
            assert result['annual_2024'] == 1.0, "Should have weight 1.0 for previous year annual"

    def test_get_weight_matrix_q2_with_q2_available(self, calculator):
        """Test get_weight_matrix for Q2 when Q2 report is available"""
        with patch('tushare_provider.update_annual_report_ttm.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 6, 15)  # June, Q2 available

            result = calculator.get_weight_matrix(2, 6)

            expected_keys = ['annual_2024', 'q1_2025', 'q2_2025']
            for key in expected_keys:
                assert key in result, f"Should include {key} when Q2 is available"

    def test_get_weight_matrix_q2_without_q2_available(self, calculator):
        """Test get_weight_matrix for Q2 when Q2 report is not available"""
        with patch('tushare_provider.update_annual_report_ttm.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 3, 15)  # March, Q2 not available

            result = calculator.get_weight_matrix(2, 3)

            expected_keys = ['annual_2024', 'q1_2025']
            for key in expected_keys:
                assert key in result, f"Should include {key} when Q2 is not available"

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
            'ts_code': ['000001.SZ'] * 3,
            'report_period': ['2025-06-30', '2024-12-31', '2024-06-30'],
            'report_year': [2025, 2024, 2024],
            'ann_date': [datetime(2025, 6, 30), datetime(2024, 12, 31), datetime(2024, 6, 30)],
            'revenue': [300, 1000, 250],
            'gross_profit_margin': [0.25, 0.30, 0.20],
            'operating_profit_margin': [0.15, 0.18, 0.12]
        })

        target_date = "20250630"  # Q2 2025

        result = calculator.calculate_ttm_metrics(df, target_date)

        # Check revenue calculation: 300 + 1000 - 250 = 1050
        assert result['revenue_ttm'] == 1050.0

        # Check margin calculations (should be calculated from the YTD formula)
        # gross_margin: 0.25 + 0.30 - 0.20 = 0.35
        assert result['gross_margin_ttm'] == 0.35

        # operating_margin: 0.15 + 0.18 - 0.12 = 0.21
        assert result['operating_margin_ttm'] == 0.21


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
