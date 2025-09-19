import os
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime
import argparse
import time
import logging
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tushare init with error handling
if "TUSHARE" not in os.environ:
    raise ValueError("TUSHARE environment variable not set. Please set your Tushare token.")
ts.set_token(os.environ["TUSHARE"])
pro = ts.pro_api()

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def retry_api_call(func, *args, **kwargs):
    """Retry API call with exponential backoff"""
    last_exception = None

    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                logger.warning(f"API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"API call failed after {MAX_RETRIES} attempts: {e}")

    # If all retries failed, raise the last exception
    raise last_exception

# Income statement fields (core primitives for calculations)
INCOME_COLUMNS = [
    "total_revenue", "revenue", "operate_profit", "total_profit", "n_income_attr_p", "basic_eps",
    "total_cogs", "oper_cost", "sell_exp", "admin_exp", "fin_exp", "invest_income", "int_exp",
    "oper_exp", "ebit", "ebitda", "income_tax", "comshare_payable_dvd", "rd_exp"
]

# Balance sheet fields (core primitives)
BALANCE_COLUMNS = [
    "total_assets", "total_liab", "total_hldr_eqy_inc_min_int", "total_cur_assets",
    "total_cur_liab", "accounts_receiv", "inventories", "acct_payable",
    "fix_assets", "lt_borr", "r_and_d", "goodwill", "intang_assets", "st_borr",
    "total_share", "oth_eqt_tools_p_shr", "total_hldr_eqy_exc_min_int",
    "money_cap"  # Currency funds for more accurate ROIC calculation
]

# Cash flow statement fields (core primitives)
CASHFLOW_COLUMNS = [
    "n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act",
    "free_cashflow", "c_pay_acq_const_fiolta", "c_fr_sale_sg", "c_paid_goods_s",
    "c_paid_to_for_empl", "c_paid_for_taxes", "n_incr_cash_cash_equ",
    "c_disp_withdrwl_invest", "c_pay_dist_dpcp_int_exp", "c_cash_equ_end_period"
]

# Financial indicator fields to validate (extended core set)
INDICATOR_COLUMNS = [
    # Basic financial indicators
    "eps", "dt_eps", "gross_margin", "netprofit_margin", "grossprofit_margin",
    "ebitda_margin", "extra_item", "profit_dedt", "op_income", "daa",

    # Solvency indicators (core)
    "current_ratio", "quick_ratio", "cash_ratio", "debt_to_assets", "assets_to_eqt",
    "dp_assets_to_eqt", "debt_to_eqt", "ebit_to_interest", "ebitda_to_debt",

    # Operating efficiency indicators (core)
    "inv_turn", "ar_turn", "ca_turn", "fa_turn", "assets_turn", "inventory_turnover",
    "currentasset_turnover", "arturnover",

    # Profitability indicators (core)
    "roic", "roe_waa", "roe_dt", "roe_yearly", "roa", "npta", "npta_yearly",
    "roa_yearly", "roa_dp",

    # Cash flow indicators (core)
    "cf_sales", "cf_nm", "cf_liabs", "cashflow_m", "ocf_to_debt",
    "ocf_to_interestdebt", "ocf_to_netdebt", "ocf_to_shortdebt", "ocf_to_or",
    "ocf_to_opincome", "salescash_to_or", "q_ocf_to_sales",

    # Growth indicators (core)
    "basic_eps_yoy", "dt_eps_yoy", "cfps_yoy", "op_yoy", "ebt_yoy", "netprofit_yoy",
    "dt_netprofit_yoy", "ocf_yoy", "roe_yoy", "bps_yoy", "assets_yoy", "eqt_yoy",
    "tr_yoy", "or_yoy", "equity_yoy",

    # Quarterly financial indicators (core)
    "q_netprofit_margin", "q_gsprofit_margin", "q_exp_to_sales", "q_profit_to_gr",
    "q_saleexp_to_gr", "q_adminexp_to_gr", "q_finaexp_to_gr", "q_roe", "q_dt_roe", "q_npta",

    # Cost and expense structure (core)
    "cogs_of_sales", "expense_of_sales", "profit_to_gr", "saleexp_to_gr",
    "adminexp_to_gr", "finaexp_to_gr", "gc_of_gr", "op_to_ebt", "tax_to_ebt",
    "dtprofit_to_profit", "profit_to_op",

    # Asset structure analysis (core)
    "ca_to_assets", "nca_to_assets", "tbassets_to_totalassets", "int_to_talcap",
    "eqt_to_talcapital", "currentdebt_to_debt", "longdeb_to_debt",
    "longdebt_to_workingcapital"
]

# API field name list (all three major financial statements contain these base fields)
# Note: API returns 'end_date' but database stores as 'ann_date'
API_COMMON_FIELDS = ['ts_code', 'ann_date', 'end_date', 'report_type']

# Financial indicators base fields (does not include report_type)
INDICATOR_BASE_FIELDS = ['ts_code', 'ann_date', 'end_date']  # Keep end_date for API call, will be mapped later

def generate_periods(start_date: str, end_date: str, period: str = "annual") -> List[str]:
    """
    Generate list of periods that need to be processed between start_date and end_date

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        period: "annual" or "quarter"

    Returns:
        List of period strings in YYYYMMDD format
    """
    periods = []
    # Support both YYYY-MM-DD and YYYYMMDD formats
    try:
        if '-' in start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD or YYYYMMDD. Error: {e}")

    if period == "annual":
        # Generate all annual report dates (12-31) within the date range
        for year in range(start_dt.year, end_dt.year + 1):
            annual_date = datetime(year, 12, 31)

            # Include if the annual date falls within our date range
            # or if it's the annual date for years that overlap with our range
            if start_dt <= annual_date <= end_dt:
                periods.append(f"{year}1231")

    elif period == "quarter":
        # Generate all quarterly report dates within the date range
        quarters = [(3, 31), (6, 30), (9, 30), (12, 31)]

        # Iterate through each year in the range
        for year in range(start_dt.year, end_dt.year + 1):
            for month, day in quarters:
                quarter_date = datetime(year, month, day)

                # Check if this quarter date falls within our date range
                if start_dt <= quarter_date <= end_dt:
                    periods.append(quarter_date.strftime("%Y%m%d"))

    else:
        raise ValueError(f"Invalid period type: {period}. Must be 'annual' or 'quarter'")

    # Sort periods chronologically
    periods.sort()

    return periods

def fetch_tushare_data(stocks: str, periods: List[str]):
    """Fetch data from multiple Tushare interfaces"""
    try:
        # Handle stocks parameter
        if stocks:
            stocks_list = [s.strip() for s in stocks.split(',') if s.strip()]
        else:
            stocks_list = []

        income_dfs = []
        balance_dfs = []
        cashflow_dfs = []
        fina_dfs = []

        if stocks_list:
            for ts_code in stocks_list:
                for period in periods:
                    try:
                        # Income statement (doc_id=33) - focus on basics
                        income_df = retry_api_call(
                            pro.income_vip,
                            ts_code=ts_code,
                            period=period,
                            fields=','.join(API_COMMON_FIELDS + INCOME_COLUMNS)
                        )
                        if not income_df.empty:
                            income_dfs.append(income_df)

                        # Balance sheet (doc_id=36) - for equity/assets
                        balance_df = retry_api_call(
                            pro.balancesheet_vip,
                            ts_code=ts_code,
                            period=period,
                            fields=','.join(API_COMMON_FIELDS + BALANCE_COLUMNS)
                        )
                        if not balance_df.empty:
                            balance_dfs.append(balance_df)

                        # Cash flow (doc_id=44) - basic for completeness
                        cashflow_df = retry_api_call(
                            pro.cashflow_vip,
                            ts_code=ts_code,
                            period=period,
                            fields=','.join(API_COMMON_FIELDS + CASHFLOW_COLUMNS)
                        )
                        if not cashflow_df.empty:
                            cashflow_dfs.append(cashflow_df)

                        # Financial indicators (doc_id=79) - for validation (extended fields)
                        indicator_fields = INDICATOR_BASE_FIELDS + INDICATOR_COLUMNS
                        fina_df = retry_api_call(
                            pro.fina_indicator_vip,
                            ts_code=ts_code,
                            period=period,
                            fields=','.join(indicator_fields)
                        )
                        if not fina_df.empty:
                            fina_dfs.append(fina_df)
                    except Exception as e:
                        print(f"Error fetching data for {ts_code} period {period} after retries: {e}")
                        continue
        else:
            # Fetch sample data for testing (limit periods to avoid rate limits)
            print("Warning: No specific stocks provided, fetching sample data for testing")
            for period in periods:  # Limit periods for testing
                try:
                    # Income statement
                    income_df = retry_api_call(
                        pro.income_vip,
                        period=period,
                        fields=','.join(API_COMMON_FIELDS + INCOME_COLUMNS)
                    )
                    if not income_df.empty:
                        income_dfs.append(income_df)

                    # Balance sheet
                    balance_df = retry_api_call(
                        pro.balancesheet_vip,
                        period=period,
                        fields=','.join(API_COMMON_FIELDS + BALANCE_COLUMNS)
                    )
                    if not balance_df.empty:
                        balance_dfs.append(balance_df)

                    # Cash flow
                    cashflow_df = retry_api_call(
                        pro.cashflow_vip,
                        period=period,
                        fields=','.join(API_COMMON_FIELDS + CASHFLOW_COLUMNS)
                    )
                    if not cashflow_df.empty:
                        cashflow_dfs.append(cashflow_df)

                    # Financial indicators
                    indicator_fields = INDICATOR_BASE_FIELDS + INDICATOR_COLUMNS
                    fina_df = retry_api_call(
                        pro.fina_indicator_vip,
                        period=period,
                        fields=','.join(indicator_fields)
                    )
                    if not fina_df.empty:
                        fina_dfs.append(fina_df)
                except Exception as e:
                    print(f"Error fetching data for period {period} after retries: {e}")
                    continue

        # Concatenate all dataframes
        income_df = pd.concat(income_dfs, ignore_index=True) if income_dfs else pd.DataFrame()
        balance_df = pd.concat(balance_dfs, ignore_index=True) if balance_dfs else pd.DataFrame()
        cashflow_df = pd.concat(cashflow_dfs, ignore_index=True) if cashflow_dfs else pd.DataFrame()
        fina_df = pd.concat(fina_dfs, ignore_index=True) if fina_dfs else pd.DataFrame()

        return income_df, balance_df, cashflow_df, fina_df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def compute_basic_indicators(income_df, balance_df, cashflow_df):
    """Compute extended basic indicators from primitives"""
    if income_df.empty or balance_df.empty or cashflow_df.empty:
        return pd.DataFrame()

    # Ensure all dataframes have report_period column
    income_df['report_period'] = income_df['end_date'].str.replace('-', '')
    balance_df['report_period'] = balance_df['end_date'].str.replace('-', '')
    cashflow_df['report_period'] = cashflow_df['end_date'].str.replace('-', '')

    # Check if we have the required columns for merge
    merge_keys = ['ts_code', 'report_period']

    # Merge dataframes
    try:
        df = pd.merge(pd.merge(income_df, balance_df, on=merge_keys, how='inner'),
                      cashflow_df, on=merge_keys, how='inner')
    except KeyError as e:
        print(f"Merge failed due to missing key: {e}")
        print(f"Available columns in income_df: {list(income_df.columns)}")
        print(f"Available columns in balance_df: {list(balance_df.columns)}")
        print(f"Available columns in cashflow_df: {list(cashflow_df.columns)}")
        return pd.DataFrame()
    df = df.copy()
    
    # 1. Gross profit and margin
    df['calc_gross_profit'] = df['revenue'] - df['oper_cost']
    df['calc_grossprofit_margin'] = np.where(
        df['revenue'] > 0, (df['calc_gross_profit'] / df['revenue']) * 100, np.nan
    )
    
    # 2. Net margin
    df['calc_netprofit_margin'] = np.where(
        df['revenue'] > 0, (df['n_income_attr_p'] / df['revenue']) * 100, np.nan
    )
    
    # 3. EBITDA margin
    df['calc_ebitda_margin'] = np.where(
        df['total_revenue'] > 0, (df['ebitda'] / df['total_revenue']) * 100, np.nan
    )
    
    # 4. Solvency
    df['calc_current_ratio'] = np.where(
        df['total_cur_liab'] > 0, df['total_cur_assets'] / df['total_cur_liab'], np.nan
    )
    df['calc_quick_ratio'] = np.where(
        df['total_cur_liab'] > 0, (df['total_cur_assets'] - df['inventories']) / df['total_cur_liab'], np.nan
    )
    df['calc_cash_ratio'] = np.where(
        df['total_cur_liab'] > 0, (df['n_cashflow_act'] / df['total_cur_liab']) * 100, np.nan  # Approx cash
    )
    df['calc_debt_to_assets'] = np.where(
        df['total_assets'] > 0, (df['total_liab'] / df['total_assets']) * 100, np.nan  # Convert to percentage
    )
    df['calc_assets_to_eqt'] = np.where(
        df['total_hldr_eqy_inc_min_int'] > 0, df['total_assets'] / df['total_hldr_eqy_inc_min_int'], np.nan
    )
    # Ensure EBIT is available (use proxy if missing)
    if 'ebit' not in df.columns or df['ebit'].isna().all():
        df['ebit'] = df['operate_profit']  # Approximation: EBIT ≈ Operating Profit

    # Ensure EBIT is properly defined (standard financial calculation)
    if 'ebit' not in df.columns or df['ebit'].isna().all():
        df['ebit'] = df['operate_profit'] + df['invest_income'].fillna(0) - df['int_exp'].fillna(0)

    # Use correct field name for interest expense
    interest_field = 'int_exp' if 'int_exp' in df.columns else 'interest_exp'
    if interest_field in df.columns:
        df['calc_ebit_to_interest'] = np.where(df[interest_field] > 0, df['ebit'] / df[interest_field], np.nan)
    else:
        df['calc_ebit_to_interest'] = np.nan
    
    # Sort for shift operations (required for all average calculations)
    df = df.sort_values(['ts_code', 'report_period'])

    # Calculate all average values first (before using them)
    # 1. Average values for turnover ratios
    df['prev_inv'] = df.groupby('ts_code')['inventories'].shift(1)
    df['avg_inv'] = (df['inventories'] + df['prev_inv'].fillna(df['inventories'])) / 2
    df['prev_ar'] = df.groupby('ts_code')['accounts_receiv'].shift(1)
    df['avg_ar'] = (df['accounts_receiv'] + df['prev_ar'].fillna(df['accounts_receiv'])) / 2

    # 2. Average values for profitability ratios
    for col in ['total_hldr_eqy_inc_min_int', 'total_assets']:
        prev_col = f'prev_{col}'
        avg_col = f'avg_{col}'
        df[prev_col] = df.groupby('ts_code')[col].shift(1)  # Period initial value
        df[avg_col] = (df[col] + df[prev_col].fillna(df[col])) / 2  # Fill first period with end value

    # 5. Operating efficiency - corrected with average values
    # Corrected turnover ratios using average values and proper formulas
    # Use total_cogs for inventory turnover (matches Tushare definition)
    df['calc_inv_turn'] = np.where(df['avg_inv'] > 0, df['total_cogs'] / df['avg_inv'], np.nan)
    df['calc_ar_turn'] = np.where(df['avg_ar'] > 0, df['revenue'] / df['avg_ar'], np.nan)
    df['calc_assets_turn'] = np.where(df['avg_total_assets'] > 0, df['revenue'] / df['avg_total_assets'], np.nan)

    # 6. Profitability - refined with standard weighted average calculations

    # Corrected ROE/ROA/NPTA using average values (standard financial calculation)
    df['calc_roe_waa'] = np.where(df['avg_total_hldr_eqy_inc_min_int'] > 0, (df['n_income_attr_p'] / df['avg_total_hldr_eqy_inc_min_int']) * 100, np.nan)
    df['calc_roa'] = np.where(df['avg_total_assets'] > 0, (df['n_income_attr_p'] / df['avg_total_assets']) * 100, np.nan)
    df['calc_npta'] = df['calc_roa']  # Equivalent to ROA
    # Calculate ROIC with proper vectorization
    # NOPAT = Operating Profit * (1 - tax_rate)
    # Invested Capital = Total Liabilities - Cash & Cash Equivalents
    tax_rate = np.where(
        df['total_profit'] > 0,
        np.where(df['income_tax'].notna(), df['income_tax'] / df['total_profit'], 0.25),
        0.25  # Default tax rate
    )
    nopat = df['operate_profit'] * (1 - tax_rate)

    # ROIC/ROCE refined with money_cap preference
    # Prefer money_cap over n_cashflow_act for more accurate cash proxy
    df['cash_proxy'] = df.get('money_cap', df['n_cashflow_act'])

    # Invested capital = Total Assets - Cash proxy
    df['invested_capital'] = df['total_assets'] - df['cash_proxy']

    # Calculate average invested capital
    df['prev_invested_cap'] = df.groupby('ts_code')['invested_capital'].shift(1)
    df['avg_invested_capital'] = (df['invested_capital'] + df['prev_invested_cap'].fillna(df['invested_capital'])) / 2

    # ROIC and ROCE calculations
    df['calc_roic'] = np.where(
        df['avg_invested_capital'] > 0,
        (nopat / df['avg_invested_capital']) * 100,
        np.nan
    )
    df['calc_roce'] = df['calc_roic']  # ROCE equivalent to ROIC in this context
    
    # 7. Cash flow ratios
    df['calc_ocf_to_debt'] = np.where(
        df['total_liab'] > 0, df['n_cashflow_act'] / df['total_liab'], np.nan
    )
    df['calc_cf_sales'] = np.where(
        df['revenue'] > 0, df['n_cashflow_act'] / df['revenue'], np.nan
    )
    
    # 8. Quarterly indicators - refined handling for quarterly vs annual data
    # Check if this is quarterly or annual data
    quarterly_endings = ['0331', '0630', '0930', '1231']
    is_quarterly_data = df['report_period'].astype(str).str.endswith(tuple(quarterly_endings)).any()

    if is_quarterly_data:
        # For quarterly data, use the same calculations
        df['calc_q_netprofit_margin'] = df['calc_netprofit_margin']
        df['calc_q_roe'] = df['calc_roe_waa']  # Annualized quarterly ROE
    else:
        # For annual data, quarterly metrics don't apply
        df['calc_q_netprofit_margin'] = np.nan
        df['calc_q_roe'] = np.nan
    
    # 9. Growth (YoY) - refined with proper periods for quarterly vs annual data
    # Sort by stock and period for proper YoY calculation
    df = df.sort_values(['ts_code', 'report_period'])

    # For quarterly data, use periods=4; for annual data, use periods=1
    quarterly_endings = ['0331', '0630', '0930', '1231']
    has_quarterly = df['report_period'].astype(str).str.endswith(tuple(quarterly_endings)).any()
    periods_for_yoy = 4 if len(df) > 4 and has_quarterly else 1

    # Calculate YoY changes by stock with proper periods (standard financial calculation)
    df['calc_netprofit_yoy'] = df.groupby('ts_code')['n_income_attr_p'].pct_change(periods=periods_for_yoy, fill_method=None) * 100
    df['calc_op_yoy'] = df.groupby('ts_code')['operate_profit'].pct_change(periods=periods_for_yoy, fill_method=None) * 100
    
    # 10. Cost structure
    df['calc_cogs_of_sales'] = np.where(
        df['revenue'] > 0, df['oper_cost'] / df['revenue'] * 100, np.nan
    )
    # Expense of sales - include more expense items for comprehensive calculation
    total_expenses = (
        df['sell_exp'].fillna(0) +
        df['admin_exp'].fillna(0) +
        df['fin_exp'].fillna(0)
    )

    # Optionally include RD expense if available (research and development)
    if 'rd_exp' in df.columns:
        total_expenses += df['rd_exp'].fillna(0)

    df['calc_expense_of_sales'] = np.where(
        df['revenue'] > 0, total_expenses / df['revenue'] * 100, np.nan
    )
    
    # 11. Asset structure
    df['calc_ca_to_assets'] = np.where(
        df['total_assets'] > 0, df['total_cur_assets'] / df['total_assets'] * 100, np.nan
    )
    df['calc_currentdebt_to_debt'] = np.where(
        df['total_liab'] > 0, df['total_cur_liab'] / df['total_liab'] * 100, np.nan
    )
    
    # Select calc columns
    calc_cols = [col for col in df.columns if col.startswith('calc_')]
    return df[['ts_code', 'report_period'] + calc_cols]

def cross_validate_indicators(computed_df, fina_df):
    """Cross-validate: computed vs API (extended)"""
    if computed_df.empty or fina_df.empty:
        return pd.DataFrame(), {}

    # Ensure fina_df has report_period column
    if not fina_df.empty and 'report_period' not in fina_df.columns and 'end_date' in fina_df.columns:
        fina_df = fina_df.copy()
        fina_df['report_period'] = fina_df['end_date'].str.replace('-', '')

    # Determine merge keys based on available columns
    merge_keys = []
    if 'ts_code' in computed_df.columns and 'ts_code' in fina_df.columns:
        merge_keys.append('ts_code')
    if 'report_period' in computed_df.columns and 'report_period' in fina_df.columns:
        merge_keys.append('report_period')
    elif 'end_date' in computed_df.columns and 'end_date' in fina_df.columns:
        merge_keys.append('end_date')

    if not merge_keys:
        print("Warning: No common merge keys found between computed and fina dataframes")
        return pd.DataFrame(), {}

    try:
        merged = pd.merge(computed_df, fina_df, on=merge_keys, how='inner', suffixes=('_calc', '_api'))
    except KeyError as e:
        print(f"Cross-validation merge failed: {e}")
        print(f"Available merge keys: {merge_keys}")
        return pd.DataFrame(), {}
    
    if merged.empty:
        return pd.DataFrame(), {}
    
    # Differences (absolute, threshold 0.01 for % fields)
    diff_map = {
        'grossprofit_margin': 'calc_grossprofit_margin',
        'netprofit_margin': 'calc_netprofit_margin',
        'ebitda_margin': 'calc_ebitda_margin',
        'current_ratio': 'calc_current_ratio',
        'quick_ratio': 'calc_quick_ratio',
        'debt_to_assets': 'calc_debt_to_assets',
        'assets_to_eqt': 'calc_assets_to_eqt',
        'ebit_to_interest': 'calc_ebit_to_interest',
        'inv_turn': 'calc_inv_turn',
        'ar_turn': 'calc_ar_turn',
        'assets_turn': 'calc_assets_turn',
        'roe_waa': 'calc_roe_waa',
        'roa': 'calc_roa',
        'npta': 'calc_npta',
        'roic': 'calc_roic',
        'ocf_to_debt': 'calc_ocf_to_debt',
        'cf_sales': 'calc_cf_sales',
        'q_netprofit_margin': 'calc_q_netprofit_margin',
        'q_roe': 'calc_q_roe',
        'netprofit_yoy': 'calc_netprofit_yoy',
        'op_yoy': 'calc_op_yoy',
        'cogs_of_sales': 'calc_cogs_of_sales',
        'expense_of_sales': 'calc_expense_of_sales',
        'ca_to_assets': 'calc_ca_to_assets',
        'currentdebt_to_debt': 'calc_currentdebt_to_debt'
    }
    
    consistency_summary = {}
    for api_col, calc_col in diff_map.items():
        if api_col in merged.columns and calc_col in merged.columns:
            # Calculate absolute difference and handle NaN values
            # Ensure both columns are properly handled for NaN values
            calc_values = merged[calc_col].fillna(0)
            api_values_raw = merged[api_col].fillna(0)
            abs_diff = abs(calc_values - api_values_raw)

            # Calculate relative difference (percentage)
            # Use absolute value of API values to avoid division by zero and handle negative values
            api_values = api_values_raw.abs()
            rel_diff = np.where(
                api_values != 0,
                (abs_diff / api_values) * 100,  # Percentage difference
                np.where(abs_diff == 0, 0, np.inf)  # Handle zero API values
            )

            merged[f'{api_col}_abs_diff'] = abs_diff
            merged[f'{api_col}_rel_diff_pct'] = rel_diff

            # Use relative difference for consistency check (more meaningful for different scales)
            consistent = rel_diff <= 5.0  # 5% relative difference threshold
            consistency_summary[api_col] = consistent.mean() * 100
    
    return merged, consistency_summary

def check_completeness(df_list):
    """Completeness: NULL %, outliers, coverage (extended)"""
    names = ['income', 'balance', 'cashflow', 'fina']
    report = {}
    total_periods = len(df_list[0]) if not df_list[0].empty else 0
    
    for name, df in zip(names, df_list):
        total_rows = len(df)
        # Calculate unique stock-period combinations for accurate coverage
        if not df.empty and 'ts_code' in df.columns:
            if name == 'fina':
                # For fina, use end_date instead of report_period
                if 'end_date' in df.columns:
                    unique_combinations = df[['ts_code', 'end_date']].drop_duplicates().shape[0]
                else:
                    unique_combinations = len(df)
            elif 'report_period' in df.columns:
                unique_combinations = df[['ts_code', 'report_period']].drop_duplicates().shape[0]
            else:
                unique_combinations = len(df)
            coverage_pct = round(unique_combinations / total_periods * 100, 2) if total_periods > 0 else 0
        else:
            coverage_pct = 0

        null_keys = {
            'income': 'total_revenue',
            'balance': 'total_hldr_eqy_inc_min_int',
            'cashflow': 'n_cashflow_act',
            'fina': 'netprofit_margin'
        }
        null_key = null_keys.get(name, 'ts_code')
        null_count = df[null_key].isnull().sum() if null_key in df.columns and not df.empty else 0
        null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
        report[name] = {'rows': total_rows, 'null_pct': round(null_pct, 2), 'coverage': coverage_pct}
    
    # Outliers (extended)
    outliers = []
    if not df_list[0].empty:
        neg_revenue = (df_list[0]['total_revenue'] < 0).sum()
        neg_equity = (df_list[1]['total_hldr_eqy_inc_min_int'] < 0).sum() if not df_list[1].empty else 0
        outliers.extend([f"Negative revenue: {neg_revenue}", f"Negative equity: {neg_equity}"])
    report['outliers'] = outliers
    report['overall_null_threshold'] = all(v['null_pct'] <= 5 for v in report.values() if isinstance(v, dict))
    
    return report

# Main function (updated for extended)
def run_validation(stocks: str, start_date: str = '20240101', end_date: str = '20250918', period: str = 'annual'):
    """Run full extended validation"""
    print(f"Starting Tushare cross-validation for {stocks} ({start_date} to {end_date})")

    # Fetch
    print("1. Fetching data...")
    periods = generate_periods(start_date, end_date, period)
    income_df, balance_df, cashflow_df, fina_df = fetch_tushare_data(stocks, periods)
    
    # Compute (extended)
    print("2. Computing extended indicators...")
    computed_df = compute_basic_indicators(income_df, balance_df, cashflow_df)
    
    # Validate (extended)
    print("3. Cross-validating consistency...")
    validation_df, consistency_summary = cross_validate_indicators(computed_df, fina_df)
    
    # Completeness
    print("4. Checking completeness...")
    completeness = check_completeness([income_df, balance_df, cashflow_df, fina_df])
    
    # Report
    print("\n=== Extended Consistency Summary (%% Consistent) ===")
    for k, v in consistency_summary.items():
        status = "PASS" if v >= 95 else "WARN"
        print(f"{k}: {v:.2f}% ({status})")
    
    print("\n=== Completeness Report ===")
    for k, v in completeness.items():
        if isinstance(v, dict):
            print(f"{k}: rows={v['rows']}, null_pct={v['null_pct']}%, coverage={v['coverage']}%")
        else:
            print(f"{k}: {v}")
    
    print("\n=== Detailed Validation (Sample) ===")
    if not validation_df.empty:
        sample_cols = ['ts_code', 'report_period', 'grossprofit_margin_abs_diff', 'netprofit_margin_abs_diff', 'current_ratio_abs_diff', 'roe_waa_abs_diff']  # Sample
        print(validation_df[sample_cols].head().to_string(index=False))
    else:
        print("No overlapping periods for validation")
    
    # Save CSVs
    if not validation_df.empty:
        stock_name = stocks.replace(',', '_') if stocks else 'sample'
        validation_df.to_csv(f'{stock_name}_extended_validation.csv', index=False)
        computed_df.to_csv(f'{stock_name}_extended_computed.csv', index=False)
        print(f"\nSaved: {stock_name}_extended_validation.csv & {stock_name}_extended_computed.csv")



# Example run and test
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate data from Tushare.")
    parser.add_argument("--start_date", type=str, help="Start date in YYYY-MM-DD or YYYYMMDD format.", default='20240101')
    parser.add_argument("--end_date", type=str, help="End date in YYYY-MM-DD or YYYYMMDD format.", default=datetime.now().strftime('%Y1231'))
    parser.add_argument("--stocks", type=str, help="Stocks to validate (comma-separated).", default='')
    parser.add_argument("--period", type=str, help="Period type: annual or quarter.", default='annual')

    args = parser.parse_args()
    result = run_validation(args.stocks, args.start_date, args.end_date, args.period)

    # Print summary
    if isinstance(result, dict) and result:
        total_stocks = len(result)
        successful_stocks = sum(1 for r in result.values() if 'error' not in r)
        success_rate = successful_stocks / total_stocks * 100 if total_stocks > 0 else 0
        print(f"Validation completed: {successful_stocks}/{total_stocks} stocks processed successfully ({success_rate:.1f}% success rate)")
    else:
        print("Validation completed: No results to display")