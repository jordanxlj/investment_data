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
                'fcf_ttm', 'fcf_margin_ttm', 'debt_to_ebitda'
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
            'fcf_margin_ttm', 'debt_to_ebitda'
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

    def test_ttm_threshold_minimum_3_quarters(self):
        """Test that TTM calculation works with minimum 3 quarters of data"""
        # Create data with exactly 3 quarters
        test_data_3q = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': ['20230331', '20230630', '20230930'],
            'n_income_attr_p': [100, 150, 180],
            'total_revenue': [1000, 1200, 1300],
            'total_assets': [10000, 10500, 10800],
            'total_hldr_eqy_exc_min_int': [8000, 8200, 8400],
            'total_share': [1000] * 3,
            'im_net_cashflow_oper_act': [200, 180, 220]
        })

        result = calculate_ttm_indicators(test_data_3q.copy())

        # Should calculate TTM for the last quarter (has 3 quarters available)
        valid_ttm = result.dropna(subset=['eps_ttm'])
        assert len(valid_ttm) >= 1, "Should calculate TTM with minimum 3 quarters"

        # Check TTM values are reasonable
        last_row = result.iloc[-1]
        expected_ttm_income = 100 + 150 + 180  # Sum of 3 quarters
        expected_eps = expected_ttm_income / 1000
        assert abs(last_row['eps_ttm'] - expected_eps) < 0.01, f"EPS TTM should be {expected_eps}, got {last_row['eps_ttm']}"

    def test_fcf_calculation_with_historical_capex(self):
        """Test improved FCF calculation using historical CapEx average"""
        test_data_fcf = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 5,
            'report_period': ['20230331', '20230630', '20230930', '20231231', '20240331'],
            'n_income_attr_p': [100, 150, 180, 200, 220],
            'total_revenue': [1000, 1200, 1300, 1400, 1500],
            'total_assets': [10000, 10500, 10800, 11200, 11500],
            'total_hldr_eqy_exc_min_int': [8000, 8200, 8400, 8600, 8800],
            'total_share': [1000] * 5,
            'im_net_cashflow_oper_act': [200, 180, 220, 250, 230],
            'n_cashflow_act': [150, 120, 180, 160, 190],  # Operating cash flow
            'c_pay_acq_const_fiolta': [50, 60, 55, 65, 58]  # CapEx
        })

        result = calculate_ttm_indicators(test_data_fcf.copy())

        # Should have FCF TTM column
        assert 'fcf_ttm' in result.columns

        # Check FCF calculation: OCF - CapEx
        # For the last row, should use historical CapEx average if available
        last_row = result.iloc[-1]
        expected_ocf = 190  # Last OCF value
        capex_avg = np.mean([50, 60, 55, 65, 58])  # Historical CapEx average
        expected_fcf = expected_ocf - capex_avg

        assert abs(last_row['fcf_ttm'] - expected_fcf) < 1.0, f"FCF should be ~{expected_fcf}, got {last_row['fcf_ttm']}"

    def test_fcf_calculation_fallback(self):
        """Test FCF calculation fallback when CapEx data is missing"""
        test_data_fcf_fallback = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': ['20230331', '20230630', '20230930'],
            'n_income_attr_p': [100, 150, 180],
            'total_revenue': [1000, 1200, 1300],
            'total_assets': [10000, 10500, 10800],
            'total_hldr_eqy_exc_min_int': [8000, 8200, 8400],
            'total_share': [1000] * 3,
            'im_net_cashflow_oper_act': [200, 180, 220],
            'n_cashflow_act': [150, 120, 180]  # Only OCF, no CapEx
        })

        result = calculate_ttm_indicators(test_data_fcf_fallback.copy())

        # Should still calculate FCF using conservative approximation
        assert 'fcf_ttm' in result.columns

        # Last row should use 70% of OCF as approximation
        last_row = result.iloc[-1]
        expected_fcf_fallback = 180 * 0.7  # 70% of last OCF

        assert abs(last_row['fcf_ttm'] - expected_fcf_fallback) < 1.0, f"FCF fallback should be ~{expected_fcf_fallback}, got {last_row['fcf_ttm']}"

    def test_debt_to_ebitda_with_net_debt(self):
        """Test improved Debt to EBITDA calculation using net debt"""
        test_data_debt = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': ['20230331', '20230630', '20230930'],
            'n_income_attr_p': [100, 150, 180],
            'total_revenue': [1000, 1200, 1300],
            'total_assets': [10000, 10500, 10800],
            'total_hldr_eqy_exc_min_int': [8000, 8200, 8400],
            'total_share': [1000] * 3,
            'im_net_cashflow_oper_act': [200, 180, 220],
            'total_liab': [2000, 2100, 2200],  # Total liabilities
            'money_cap': [500, 550, 600],      # Cash and equivalents
            'ebitda': [300, 320, 340]          # EBITDA
        })

        result = calculate_ttm_indicators(test_data_debt.copy())

        # Should have debt_to_ebitda column
        assert 'debt_to_ebitda' in result.columns

        # Check net debt calculation: total_liab - money_cap
        last_row = result.iloc[-1]
        net_debt = 2200 - 600  # 1600
        expected_ratio = net_debt / 340  # ~4.71

        assert abs(last_row['debt_to_ebitda'] - expected_ratio) < 0.01, f"Debt/EBITDA should be ~{expected_ratio}, got {last_row['debt_to_ebitda']}"

    def test_debt_to_ebitda_fallback_to_total_liab(self):
        """Test Debt to EBITDA fallback when cash data is missing"""
        test_data_debt_fallback = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 3,
            'report_period': ['20230331', '20230630', '20230930'],
            'n_income_attr_p': [100, 150, 180],
            'total_revenue': [1000, 1200, 1300],
            'total_assets': [10000, 10500, 10800],
            'total_hldr_eqy_exc_min_int': [8000, 8200, 8400],
            'total_share': [1000] * 3,
            'im_net_cashflow_oper_act': [200, 180, 220],
            'total_liab': [2000, 2100, 2200],  # Total liabilities only
            'ebitda': [300, 320, 340]          # EBITDA
        })

        result = calculate_ttm_indicators(test_data_debt_fallback.copy())

        # Should still calculate using total liabilities as fallback
        assert 'debt_to_ebitda' in result.columns

        last_row = result.iloc[-1]
        expected_ratio_fallback = 2200 / 340  # ~6.47 (higher than net debt ratio)

        assert abs(last_row['debt_to_ebitda'] - expected_ratio_fallback) < 0.01, f"Debt/EBITDA fallback should be ~{expected_ratio_fallback}, got {last_row['debt_to_ebitda']}"

    def test_yuan_to_wan_fields_cleanup(self):
        """Test that YUAN_TO_WAN_FIELDS only contains fields defined in DDL"""
        from src.tushare_provider.update_a_stock_financial_profile import YUAN_TO_WAN_FIELDS

        # Check that all fields in YUAN_TO_WAN_FIELDS are actually monetary fields that need conversion
        monetary_indicators = ['rd_exp']  # Only rd_exp from financial indicators should be in this list

        # All fields should be in the main data source categories (income, balance, cashflow)
        # or the specific monetary indicators
        for field in YUAN_TO_WAN_FIELDS:
            # rd_exp is the only financial indicator that should be converted
            if field not in monetary_indicators:
                # Should be in one of the main financial statement categories
                is_income_field = field in [
                    'basic_eps', 'diluted_eps', 'total_revenue', 'revenue',
                    'total_cogs', 'oper_cost', 'sell_exp', 'admin_exp', 'fin_exp',
                    'assets_impair_loss', 'operate_profit', 'non_oper_income', 'non_oper_exp',
                    'total_profit', 'income_tax', 'n_income', 'n_income_attr_p', 'ebit',
                    'ebitda', 'invest_income', 'interest_exp', 'oper_exp', 'comshare_payable_dvd'
                ]
                is_balance_field = field in [
                    'total_share', 'cap_rese', 'undistr_porfit', 'surplus_rese', 'money_cap',
                    'accounts_receiv', 'oth_receiv', 'prepayment', 'inventories',
                    'oth_cur_assets', 'total_cur_assets', 'htm_invest', 'fix_assets',
                    'intan_assets', 'defer_tax_assets', 'total_nca', 'total_assets',
                    'acct_payable', 'payroll_payable', 'taxes_payable', 'oth_payable',
                    'total_cur_liab', 'defer_inc_non_cur_liab', 'total_ncl', 'total_liab',
                    'total_hldr_eqy_exc_min_int', 'total_hldr_eqy_inc_min_int',
                    'total_liab_hldr_eqy', 'oth_pay_total', 'accounts_receiv_bill',
                    'accounts_pay', 'oth_rcv_total', 'fix_assets_total', 'lt_borr', 'st_borr',
                    'oth_eqt_tools_p_shr', 'r_and_d', 'goodwill'
                ]
                is_cashflow_field = field in [
                    'net_profit', 'finan_exp', 'c_fr_sale_sg', 'c_inf_fr_operate_a',
                    'c_paid_goods_s', 'c_paid_to_for_empl', 'c_paid_for_taxes',
                    'n_cashflow_act', 'n_cashflow_inv_act', 'free_cashflow',
                    'n_cash_flows_fnc_act', 'n_incr_cash_cash_equ', 'c_cash_equ_beg_period',
                    'c_cash_equ_end_period', 'im_net_cashflow_oper_act', 'end_bal_cash',
                    'beg_bal_cash', 'c_pay_acq_const_fiolta', 'c_disp_withdrwl_invest',
                    'c_pay_dist_dpcp_int_exp'
                ]

                assert is_income_field or is_balance_field or is_cashflow_field, \
                    f"Field {field} in YUAN_TO_WAN_FIELDS is not in any expected category"


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

    @patch('src.tushare_provider.update_a_stock_financial_profile.ts')
    @patch('src.tushare_provider.update_a_stock_financial_profile.create_engine')
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

    @patch('src.tushare_provider.update_a_stock_financial_profile.create_engine')
    def test_global_error_handling_database_error(self, mock_create_engine):
        """Test that database connection errors are properly handled"""
        # Mock database engine to raise an exception
        mock_create_engine.side_effect = Exception("Database connection failed")

        # Should raise the exception (not swallow it)
        with pytest.raises(Exception, match="Database connection failed"):
            update_a_stock_financial_profile(
                mysql_url="mysql+pymysql://invalid:invalid@invalid:3306/invalid",
                end_date="20231231",
                period="quarter",
                limit=1
            )

    @patch('src.tushare_provider.update_a_stock_financial_profile._fetch_single_period_data')
    @patch('src.tushare_provider.update_a_stock_financial_profile.create_engine')
    def test_global_error_handling_api_error(self, mock_create_engine, mock_fetch):
        """Test that API errors during data fetching are properly handled"""
        # Mock successful database connection
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Mock API call to raise an exception
        mock_fetch.side_effect = Exception("API call failed")

        # Should raise the exception (not swallow it)
        with pytest.raises(Exception, match="API call failed"):
            update_a_stock_financial_profile(
                mysql_url="mysql+pymysql://test:test@localhost:3306/test",
                end_date="20231231",
                period="quarter",
                limit=1
            )


if __name__ == "__main__":
    pytest.main([__file__])
