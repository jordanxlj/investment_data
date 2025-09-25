import pytest
import pandas as pd
import numpy as np
import warnings
import logging
from unittest.mock import patch
import io
import sys
from unittest.mock import patch, MagicMock

from src.tushare_provider.update_a_stock_financial_profile import (
    calculate_ttm_indicators,
    TTM_COLUMNS,
    ALL_COLUMNS,
    _coerce_schema,
    _fetch_single_period_data,
    _generate_periods,
    update_a_stock_financial_profile
)


class TestTTMCalculation:
    """Test TTM (Trailing Twelve Months) indicator calculations"""

    def setup_method(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 5,
            'report_period': ['20230331', '20230630', '20230930', '20231231', '20240331'],
            'n_income_attr_p': [100, 150, 120, 180, 200],  # Quarterly net income
            'total_revenue': [1000, 1200, 1100, 1300, 1400],  # Quarterly revenue
            'total_assets': [10000, 10500, 10800, 11200, 11500],  # Total assets
            'total_hldr_eqy_exc_min_int': [8000, 8200, 8400, 8600, 8800],  # Equity
            'total_share': [1000] * 5,  # Total shares
            'im_net_cashflow_oper_act': [200, 180, 220, 250, 230]  # Operating cash flow
        })

    def test_calculate_ttm_indicators_basic(self):
        """Test basic TTM calculation functionality"""
        # Test with warnings as errors to catch FutureWarnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=FutureWarning)

            result = calculate_ttm_indicators(self.test_data.copy())

            # Verify result is DataFrame
            assert isinstance(result, pd.DataFrame)
            assert not result.empty

            # Verify TTM columns exist (including new efficiency indicators)
            expected_ttm_cols = [
                'eps_ttm', 'revenue_ps_ttm', 'ocfps_ttm', 'cfps_ttm',
                'roe_ttm', 'roa_ttm', 'netprofit_margin_ttm', 'grossprofit_margin_ttm',
                'fcf_ttm', 'fcf_margin_ttm', 'debt_to_ebitda_ttm'
            ]

            existing_cols = [col for col in expected_ttm_cols if col in result.columns]
            assert len(existing_cols) >= 8, f"Should have at least 8 TTM columns, got {len(existing_cols)}"
            print(f"✅ TTM columns created: {len(existing_cols)}/{len(expected_ttm_cols)}")

    def test_calculate_ttm_indicators_values(self):
        """Test TTM calculation produces reasonable values"""
        result = calculate_ttm_indicators(self.test_data.copy())

        # Check that we have some valid TTM values (should have values for last 2 quarters)
        valid_rows = result.dropna(subset=['eps_ttm'])
        assert len(valid_rows) >= 2, "Should have at least 2 valid TTM calculations"

        # Check EPS TTM calculation (net income / shares)
        # For the last quarter: (100+150+120+180)/1000 = 0.55, but wait...
        # Actually the rolling sum should be 4 quarters: 150+120+180+200 = 650 / 1000 = 0.65
        last_row = result.iloc[-1]
        assert last_row['eps_ttm'] > 0, "EPS TTM should be positive"

        # Check ROE TTM (net income / equity)
        assert last_row['roe_ttm'] > 0, "ROE TTM should be positive"

    def test_calculate_ttm_indicators_growth(self):
        """Test CAGR calculations"""
        result = calculate_ttm_indicators(self.test_data.copy())

        # Should have CAGR columns
        assert 'revenue_cagr_3y' in result.columns
        assert 'netincome_cagr_3y' in result.columns

    def test_ttm_columns_definition(self):
        """Test TTM columns are properly defined"""
        expected_cols = [
            'eps_ttm', 'revenue_ps_ttm', 'cfps_ttm',
            'roe_ttm', 'roa_ttm', 'netprofit_margin_ttm', 'grossprofit_margin_ttm',
            'revenue_cagr_3y', 'netincome_cagr_3y',
            'fcf_margin_ttm', 'debt_to_ebitda_ttm'
        ]

        assert len(TTM_COLUMNS) == len(expected_cols)
        for col in expected_cols:
            assert col in TTM_COLUMNS

    def test_missing_data_handling_intermediate(self, caplog):
        """Test handling of intermediate missing data (gaps in time series)"""

        # Create data with intermediate missing quarters
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 4,
            'report_period': ['20230331', '20230630', '20231231', '20240331'],  # Missing Q3 2023
            'n_income_attr_p': [100, 150, 180, 200],  # Q1, Q2, Q4, Q1(next year)
            'total_revenue': [1000, 1200, 1300, 1400],
            'total_assets': [10000, 10500, 11200, 11500],
            'total_hldr_eqy_exc_min_int': [8000, 8200, 8600, 8800],
            'total_share': [1000] * 4,
            'im_net_cashflow_oper_act': [200, 180, 250, 230]
        })

        # Capture log output to check for missing data warnings
        with caplog.at_level(logging.WARNING):
            result = calculate_ttm_indicators(test_data.copy())

        # Check that warning was logged about intermediate missing data
        assert any('中间数据缺失' in record.message for record in caplog.records), "Should report intermediate missing data"
        assert any('20230930' in record.message for record in caplog.records), "Should identify the missing quarter"

        # Should still calculate TTM values for available data
        valid_ttm = result.dropna(subset=['eps_ttm'])
        assert len(valid_ttm) >= 2, "Should calculate TTM for available periods"

        # Final result should not contain filled missing rows
        assert len(result) == 4, "Should keep only original data rows"

    def test_missing_data_handling_edge(self, caplog):
        """Test handling of edge missing data (data outside available range)"""

        # Create data that starts from a later quarter but has enough data for TTM
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 5,
            'report_period': ['20230930', '20231231', '20240331', '20240630', '20240930'],  # Starts from Q3 2023
            'n_income_attr_p': [120, 180, 200, 220, 240],  # 5 quarters of data
            'total_revenue': [1100, 1300, 1400, 1500, 1600],
            'total_assets': [10800, 11200, 11500, 11800, 12000],
            'total_hldr_eqy_exc_min_int': [8400, 8600, 8800, 9000, 9200],
            'total_share': [1000] * 5,
            'im_net_cashflow_oper_act': [220, 250, 230, 260, 280]
        })

        # Capture log output to check for missing data warnings
        with caplog.at_level(logging.WARNING):
            result = calculate_ttm_indicators(test_data.copy())

        # Should complete successfully and calculate TTM for available data
        valid_ttm = result.dropna(subset=['eps_ttm'])
        assert len(valid_ttm) >= 2, "Should calculate TTM for available periods with 5 quarters of data"

        # Edge missing data should not trigger intermediate missing warnings
        # (since all data within the available range is consecutive)
        assert not any('中间数据缺失' in record.message for record in caplog.records), "Should not report intermediate missing for consecutive data"

    def test_missing_data_handling_insufficient(self):
        """Test handling of insufficient data for TTM calculation"""
        # Create data with only one quarter
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 1,
            'report_period': ['20231231'],
            'n_income_attr_p': [180],
            'total_revenue': [1300],
            'total_assets': [11200],
            'total_hldr_eqy_exc_min_int': [8600],
            'total_share': [1000],
            'im_net_cashflow_oper_act': [250]
        })

        result = calculate_ttm_indicators(test_data.copy())

        # Should mark as insufficient data
        assert result['missing_type'].iloc[0] == 'insufficient_data'

        # TTM values should be NaN due to insufficient history
        assert pd.isna(result['eps_ttm'].iloc[0])

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames"""
        empty_df = pd.DataFrame()
        result = calculate_ttm_indicators(empty_df)
        assert result.empty

    def test_single_row_handling(self):
        """Test handling of single row (insufficient data for TTM)"""
        single_row = self.test_data.head(1).copy()
        result = calculate_ttm_indicators(single_row)

        # Should still return data but TTM values should be NaN
        assert not result.empty
        assert pd.isna(result['eps_ttm'].iloc[0])


class TestSchemaCoercion:
    """Test data schema coercion and validation"""

    def test_coerce_schema_basic(self):
        """Test basic schema coercion"""
        test_df = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'report_period': ['20231231'],
            'ann_date': ['20240101'],
            'period': ['annual'],
            'currency': ['CNY'],
            'total_revenue': [1000.0],
            'eps_ttm': [1.5],
            'roe': [15.5]  # Add a unique field to avoid conflicts
        })

        result = _coerce_schema(test_df)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert result['ts_code'].iloc[0] == '000001.SZ'

    def test_all_columns_exist(self):
        """Test that ALL_COLUMNS includes all expected columns"""
        assert isinstance(ALL_COLUMNS, list)
        assert len(ALL_COLUMNS) > 50  # Should have many columns

        # Check that basic columns are present
        basic_cols = ['ts_code', 'report_period', 'ann_date', 'period', 'currency']
        for col in basic_cols:
            assert col in ALL_COLUMNS

        # Check that TTM columns are included
        for col in TTM_COLUMNS:
            assert col in ALL_COLUMNS


class TestPeriodGeneration:
    """Test period generation logic"""

    def test_generate_periods_annual(self):
        """Test annual period generation"""
        periods = _generate_periods('20231231', 'annual', 3)
        assert isinstance(periods, list)
        assert len(periods) <= 3

        # Should be in YYYYMMDD format
        for period in periods:
            assert len(period) == 8
            assert period.endswith('1231')  # Annual periods end with Dec 31

    def test_generate_periods_quarterly(self):
        """Test quarterly period generation"""
        periods = _generate_periods('20231231', 'quarter', 4)
        assert isinstance(periods, list)
        assert len(periods) <= 4

        # Should be in YYYYMMDD format
        for period in periods:
            assert len(period) == 8
            # Should end with quarter end dates
            day = period[-2:]
            assert day in ['31', '30']


@pytest.mark.integration
class TestIntegration:
    """Integration tests (may require database connection)"""

    @patch('tushare_provider.update_a_stock_financial_profile.ts')
    @patch('tushare_provider.update_a_stock_financial_profile.create_engine')
    def test_update_function_structure(self, mock_create_engine, mock_ts):
        """Test that the main update function has correct structure"""
        # Mock the database engine
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Mock Tushare API
        mock_pro = MagicMock()
        mock_ts.set_token.return_value = None
        mock_ts.pro_api.return_value = mock_pro

        # This would normally try to connect to database and fetch data
        # For now, just test that the function exists and is callable
        assert callable(update_a_stock_financial_profile)


if __name__ == "__main__":
    pytest.main([__file__])
