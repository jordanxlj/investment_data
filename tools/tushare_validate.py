import os
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime
import argparse
import time
import logging
from typing import List
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
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

# Cache configuration
CACHE_DIR = "cache"
USE_CACHE = True  # Set to False to disable caching

def get_cache_path(cache_file: str) -> str:
    """Get the full path for a cache file"""
    return os.path.join(CACHE_DIR, f"{cache_file}.parquet")

def load_from_cache(cache_file: str) -> pd.DataFrame:
    """Load data from cache if it exists"""
    cache_path = get_cache_path(cache_file)
    if os.path.exists(cache_path):
        try:
            logger.info(f"Loading {cache_file} from cache: {cache_path}")
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file}: {e}")
    return pd.DataFrame()

def save_to_cache(file_name: str, df: pd.DataFrame):
    """Save data to cache"""
    if df.empty:
        return

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_path = get_cache_path(file_name)
    try:
        df.to_parquet(cache_path, index=False)
        logger.info(f"Saved to cache: {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_path}: {e}")

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
    'basic_eps', 'diluted_eps', 'total_revenue', 'revenue', 'prem_earned', 
    'ass_invest_income', 'total_cogs', 'oper_cost', 'biz_tax_surchg', 
    'sell_exp', 'admin_exp', 'fin_exp', 'assets_impair_loss', 'operate_profit',
    'non_oper_income', 'non_oper_exp', 'total_profit', 'income_tax', 'n_income', 
    'n_income_attr_p', 'oth_compr_income', 't_compr_income', 'compr_inc_attr_p', 
    'ebit', 'ebitda', 'total_opcost', 'invest_income'
]

# Balance sheet fields (core primitives)
BALANCE_COLUMNS = [
    'total_share', 'cap_rese', 'undistr_porfit', 'surplus_rese', 'money_cap', 
    'accounts_receiv', 'oth_receiv', 'prepayment', 'inventories', 'premium_receiv', 
    'reinsur_receiv', 'oth_cur_assets', 'total_cur_assets', 'htm_invest', 
    'fix_assets', 'oil_and_gas_assets', 'intan_assets', 'defer_tax_assets', 
    'total_nca', 'cash_reser_cb', 'depos_in_oth_bfi', 'total_assets', 
    'acct_payable', 'payroll_payable', 'taxes_payable', 'oth_payable', 
    'total_cur_liab', 'defer_inc_non_cur_liab', 'total_ncl',  
    'total_liab', 'total_hldr_eqy_exc_min_int', 'total_hldr_eqy_inc_min_int', 
    'total_liab_hldr_eqy', 'oth_pay_total', 'accounts_receiv_bill', 
    'accounts_pay', 'oth_rcv_total', 'fix_assets_total'
]

# Cash flow statement fields (core primitives)
CASHFLOW_COLUMNS = [
    'net_profit', 'finan_exp', 'c_fr_sale_sg', 
    'c_fr_oth_operate_a', 'c_inf_fr_operate_a', 'c_paid_goods_s', 'c_paid_to_for_empl', 
    'c_paid_for_taxes', 'oth_cash_pay_oper_act', 'st_cash_out_act', 'n_cashflow_act', 
    'stot_inflows_inv_act', 'c_pay_acq_const_fiolta', 'stot_out_inv_act', 
    'n_cashflow_inv_act', 'oth_cash_recp_ral_fnc_act', 'stot_cash_in_fnc_act', 
    'free_cashflow', 'c_pay_dist_dpcp_int_exp', 'stot_cashout_fnc_act', 'n_cash_flows_fnc_act', 
    'n_incr_cash_cash_equ', 'c_cash_equ_beg_period', 'c_cash_equ_end_period', 
    'c_recp_cap_contrib', 'prov_depr_assets', 'depr_fa_coga_dpba', 'amort_intang_assets', 
    'decr_def_inc_tax_assets', 'decr_inventories', 'decr_oper_payable', 'incr_oper_payable', 
    'im_net_cashflow_oper_act', 'im_n_incr_cash_equ', 'net_dism_capital_add', 
    'credit_impa_loss', 'end_bal_cash', 'beg_bal_cash'
]

# Financial indicator fields to validate (extended core set)
INDICATOR_COLUMNS = [
    'eps', 'dt_eps', 'total_revenue_ps', 'revenue_ps', 'capital_rese_ps', 'surplus_rese_ps', 
    'undist_profit_ps', 'extra_item', 'profit_dedt', 'gross_margin', 'current_ratio', 
    'quick_ratio', 'cash_ratio', 'invturn_days', 'arturn_days', 'inv_turn', 'ar_turn', 
    'ca_turn', 'fa_turn', 'assets_turn', 'op_income', 'valuechange_income', 'interst_income', 
    'daa', 'fcff', 'fcfe', 'current_exint', 'noncurrent_exint', 
    'interestdebt', 'netdebt', 'tangible_asset', 'working_capital', 'networking_capital', 
    'invest_capital', 'retained_earnings', 'diluted2_eps', 'bps', 'ocfps', 'retainedps', 
    'cfps', 'ebit_ps', 'fcff_ps', 'fcfe_ps', 'netprofit_margin', 'grossprofit_margin', 
    'cogs_of_sales', 'expense_of_sales', 'profit_to_gr', 'saleexp_to_gr', 'adminexp_of_gr', 
    'finaexp_of_gr', 'impai_ttm', 'gc_of_gr', 'op_of_gr', 'ebit_of_gr', 'roe', 'roe_waa', 
    'roe_dt', 'roa', 'npta', 'roic', 'roe_yearly', 'roa2_yearly', 'roe_avg', 'salescash_to_or', 
    'ocf_to_or', 'capitalized_to_da', 'debt_to_assets', 'assets_to_eqt', 'dp_assets_to_eqt', 
    'ca_to_assets', 'nca_to_assets', 'tbassets_to_totalassets', 'int_to_talcap', 
    'eqt_to_talcapital', 'currentdebt_to_debt', 'longdeb_to_debt', 'ocf_to_shortdebt', 
    'debt_to_eqt', 'eqt_to_debt', 'tangibleasset_to_debt', 'ocf_to_debt', 'ebitda_to_debt', 
    'turn_days', 'roa_yearly', 'roa_dp', 'fixed_assets', 'profit_prefin_exp', 'non_op_profit', 
    'cash_to_liqdebt', 'op_to_liqdebt', 'op_to_debt', 'roic_yearly', 'total_fa_trun', 
    'profit_to_op', 'q_investincome_to_ebt', 'basic_eps_yoy', 'dt_eps_yoy', 'cfps_yoy', 
    'op_yoy', 'ebt_yoy', 'netprofit_yoy', 'dt_netprofit_yoy', 'ocf_yoy', 'roe_yoy', 
    'bps_yoy', 'assets_yoy', 'eqt_yoy', 'tr_yoy', 'or_yoy', 'equity_yoy', 'rd_exp'
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

def fetch_tushare_data_from_api(stocks: str, periods: List[str]):
    """Fetch data from multiple Tushare interfaces (internal function without caching)"""
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

def fetch_tushare_data(stocks: str, periods: List[str]):
    """Fetch data from multiple Tushare interfaces with caching support"""
    if not USE_CACHE:
        logger.info("Cache disabled, fetching data from Tushare API")
        return fetch_tushare_data_from_api(stocks, periods)

    # Try to load from cache first
    income_df = load_from_cache('income_data')
    balance_df = load_from_cache('balance_data')
    cashflow_df = load_from_cache('cashflow_data')
    fina_df = load_from_cache('fina_data')

    # Check if all data is available in cache
    all_cached = not (income_df.empty or balance_df.empty or cashflow_df.empty or fina_df.empty)

    if all_cached:
        logger.info("All data loaded from cache successfully")
        return income_df, balance_df, cashflow_df, fina_df
    else:
        logger.info("Some data missing from cache, fetching from Tushare API")

        # Fetch missing data from API
        api_income_df, api_balance_df, api_cashflow_df, api_fina_df = fetch_tushare_data_from_api(stocks, periods)

        # Update cache with API data
        if not api_income_df.empty:
            save_to_cache("income_data", api_income_df)
            if income_df.empty:
                income_df = api_income_df

        if not api_balance_df.empty:
            save_to_cache("balance_data", api_balance_df)
            if balance_df.empty:
                balance_df = api_balance_df

        if not api_cashflow_df.empty:
            save_to_cache("cashflow_data", api_cashflow_df)
            if cashflow_df.empty:
                cashflow_df = api_cashflow_df

        if not api_fina_df.empty:
            save_to_cache("fina_data", api_fina_df)
            if fina_df.empty:
                fina_df = api_fina_df

        return income_df, balance_df, cashflow_df, fina_df

def compute_basic_indicators(income_df, balance_df, cashflow_df, fina_df):
    """Compute extended basic indicators from primitives"""
    if income_df.empty or balance_df.empty or cashflow_df.empty:
        return pd.DataFrame()

    # Ensure all dataframes have report_period column
    income_df['report_period'] = income_df['end_date'].str.replace('-', '')
    balance_df['report_period'] = balance_df['end_date'].str.replace('-', '')
    cashflow_df['report_period'] = cashflow_df['end_date'].str.replace('-', '')
    fina_df['report_period'] = fina_df['end_date'].str.replace('-', '')

    # Check if we have the required columns for merge
    common_non_keys = ['ann_date', 'end_date', 'report_type']
    balance_df = balance_df.drop(columns=[col for col in common_non_keys if col in balance_df.columns], errors='ignore')
    cashflow_df = cashflow_df.drop(columns=[col for col in common_non_keys if col in cashflow_df.columns], errors='ignore')
    fina_df = fina_df.drop(columns=[col for col in common_non_keys if col in fina_df.columns], errors='ignore')

    # Merge dataframes with custom suffixes to avoid column conflicts
    try:
        merge_keys = ['ts_code', 'report_period']
        # First merge: income + balance
        df = pd.merge(income_df, balance_df, on=merge_keys, how='inner', suffixes=('', ''))
        # Second merge: + cashflow
        df = pd.merge(df, cashflow_df, on=merge_keys, how='inner', suffixes=('', ''))
        # Third merge: + fina indicators
        df = pd.merge(df, fina_df, on=merge_keys, how='inner', suffixes=('', ''))
    except KeyError as e:
        print(f"Merge failed due to missing key: {e}")
        print(f"Available columns in income_df: {list(income_df.columns)}")
        print(f"Available columns in balance_df: {list(balance_df.columns)}")
        print(f"Available columns in cashflow_df: {list(cashflow_df.columns)}")
        print(f"Available columns in fina_df: {list(fina_df.columns)}")
        return pd.DataFrame()
        df = df.copy()

        # Debug: Print available columns for key calculations
        logger.debug(f"Available columns in merged df: {len(df.columns)} total")
        cash_fields = [col for col in df.columns if 'cash' in col.lower() or 'money' in col.lower() or 'trad' in col.lower()]
        interest_fields = [col for col in df.columns if 'int' in col.lower() or 'fin' in col.lower() or 'interest' in col.lower()]
        logger.debug(f"Cash-related fields available: {cash_fields}")
        logger.debug(f"Interest-related fields available: {interest_fields}")

    for field in df.columns:
        field_null_count = df[field].isna().sum()
        logger.debug(f"{field}_null_pct: {field_null_count / len(df) * 100.0}")

    # 1. Gross profit and margin
    # Use total_revenue if revenue is not available
    revenue_field = 'revenue' if 'revenue' in df.columns and not df['revenue'].isna().all() else 'total_revenue'
    df['calc_gross_profit'] = df[revenue_field] - df['oper_cost']
    df['calc_grossprofit_margin'] = np.where(
        df[revenue_field] > 0, (df['calc_gross_profit'] / df[revenue_field]) * 100, np.nan
    )

    # 2. Net margin
    # Use total_revenue if revenue is not available or use revenue as primary
    revenue_field = 'revenue' if 'revenue' in df.columns and not df['revenue'].isna().all() else 'total_revenue'
    df['calc_netprofit_margin'] = np.where(
        df[revenue_field] > 0, (df['n_income_attr_p'] / df[revenue_field]) * 100, np.nan
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
    df['calc_debt_to_assets'] = np.where(
        df['total_assets'] > 0, (df['total_liab'] / df['total_assets']) * 100, np.nan  # Convert to percentage
    )
    df['calc_assets_to_eqt'] = np.where(
        df['total_hldr_eqy_inc_min_int'] > 0, df['total_assets'] / df['total_hldr_eqy_inc_min_int'], np.nan
    )
    
    # Cash ratio - since money_cap field may not be available, use alternative approach
    # In Tushare, cash ratio may be calculated differently or use different fields
    # For now, if we can't find the cash field, we'll leave it as NaN and let validation show the discrepancy
    cash_equivalent = np.nan

    # Try to find cash-related fields - expanded list
    cash_fields = ['money_cap']
    for field in cash_fields:
        if field in df.columns:
            cash_equivalent = df[field].fillna(0)
            break

    # Calculate cash ratio if we have cash data
    valid_liab = df['total_cur_liab'] > 0
    if not cash_equivalent.isna().all():
        df['calc_cash_ratio'] = np.where(valid_liab,
                                       cash_equivalent / df['total_cur_liab'], np.nan)
    else:
        # If no cash field available, set to NaN (will show as discrepancy in validation)
        df['calc_cash_ratio'] = np.nan

    # EBIT to Interest ratio - improved calculation
    # EBIT = Operating Profit + Investment Income - Interest Expense
    df['calc_ebit_full'] = (
        df['operate_profit'].fillna(0) +
        df['invest_income'].fillna(0) -
        df['fin_exp'].fillna(0)
    )

    # EBIT to Interest ratio - use fin_exp as interest expense
    df['calc_ebit_to_interest'] = np.where(
        df['fin_exp'].abs() > 0,
        df['calc_ebit_full'] / df['fin_exp'].abs(),
        np.nan
    )
    calc_ebit_to_interest_null_count = df['calc_ebit_to_interest'].isna().sum()
    logger.debug(f"calc_ebit_to_interest_null_pct: {calc_ebit_to_interest_null_count / len(df) * 100.0}")
    
    # Sort for shift operations (required for all average calculations)
    df = df.sort_values(['ts_code', 'report_period'])

    # Calculate all average values first (before using them)
    # 1. Average values for turnover ratios
    df['prev_inv'] = df.groupby('ts_code')['inventories'].shift(1)
    df['avg_inv'] = (df['inventories'] + df['prev_inv'].fillna(df['inventories'])) / 2
    df['prev_ar'] = df.groupby('ts_code')['accounts_receiv'].shift(1)
    df['avg_ar'] = (df['accounts_receiv'] + df['prev_ar'].fillna(df['accounts_receiv'])) / 2
    # Interest-Bearing Debt
    # Non-Interest Current Liabilities (exclude interest_payable)
    non_int_cur_liab = df.get('acct_payable', 0) + df.get('adv_receipts', 0) + df.get('contract_liab', 0) + df.get('payroll_payable', 0) + df.get('taxes_payable', 0) + df.get('div_payable', 0) + df.get('oth_payable', 0) + df.get('oth_cur_liab', 0)
    # Non-Interest Non-Current Liabilities
    non_int_non_cur_liab = df['total_ncl'] - df.get('lt_borr', 0) - df.get('bonds_payable', 0)
    df['int_debt'] = df['total_liab'] - non_int_cur_liab - non_int_non_cur_liab

    # 2. Average values for profitability ratios
    for col in ['total_hldr_eqy_inc_min_int', 'total_assets', 'fix_assets', 'total_cur_assets', 'int_debt']:
        prev_col = f'prev_{col}'
        avg_col = f'avg_{col}'
        df[prev_col] = df.groupby('ts_code')[col].shift(1)  # Period initial value
        df[avg_col] = (df[col] + df[prev_col].fillna(df[col])) / 2  # Fill first period with end value

    # 5. Operating efficiency - refined turnover ratios with better field mapping
    # Use oper_cost for inventory turnover (matches Tushare's inventory_turnover calculation)
    # Tushare uses oper_cost for cost of goods sold in turnover calculations
    # Try different cost fields for inventory turnover
    cost_field = 'oper_cost' if 'oper_cost' in df.columns else 'total_cogs'
    df['calc_inv_turn'] = np.where(df['avg_inv'] > 0, df[cost_field] / df['avg_inv'], np.nan)

    # Inventory Turnover Days = 360 / Inventory Turnover Rate
    # 存货周转天数 = 360 / (期末.营业成本 / ((期末.存货 + 期初.存货) / 2))
    df['calc_inv_turn_days'] = np.where(df['calc_inv_turn'] > 0, 360 / df['calc_inv_turn'], np.nan)

    # Use consistent revenue field for all turnover calculations
    revenue_field = 'revenue' if 'revenue' in df.columns and not df['revenue'].isna().all() else 'total_revenue'
    df['calc_ar_turn'] = np.where(df['avg_ar'] > 0, df[revenue_field] / df['avg_ar'], np.nan)
    df['calc_assets_turn'] = np.where(df['avg_total_assets'] > 0, df[revenue_field] / df['avg_total_assets'], np.nan)

    # Ensure we have valid equity and assets values before calculating ratios
    # Use average equity for ROE calculation (more accurate than end-of-period)
    df['calc_roe_waa'] = np.where(
        df['avg_total_hldr_eqy_inc_min_int'] > 0,
        (df['n_income_attr_p'] / df['avg_total_hldr_eqy_inc_min_int']) * 100,
        np.nan
    )

    # Use average total assets for ROA calculation
    df['calc_roa'] = np.where(
        df['avg_total_assets'] > 0,
        (df['n_income_attr_p'] / np.abs(df['avg_total_assets'])) * 100,
        np.nan
    )

    df['calc_npta'] = df['calc_roa']  # NPTA is equivalent to ROA

    # 优化tax_rate: 优先total_profit，fallback到operate_profit
    tax_rate_base = df['total_profit'].abs().fillna(df['operate_profit'].abs().fillna(0))
    tax_rate = np.where(
        tax_rate_base > 0,
        np.clip(df['income_tax'].fillna(0) / tax_rate_base, 0, 0.5),
        0.25  # 默认
    )

    # EBIT不变，但填充NaN
    df['calc_ebit'] = df.get('ebit', (df['operate_profit'] + df['invest_income'] - df['fin_exp'])).fillna(0)
    # NOPAT
    nopat = df['calc_ebit'] * (1 - tax_rate)

    # 优化 invested_capital：扣除非利息负债（更标准）
    non_interest_liab = df.get('acct_payable', 0) + df.get('taxes_payable', 0) + df.get('oth_payable', 0)
    invested_capital = (df['total_liab'] + df['total_hldr_eqy_inc_min_int'] - df.get('money_cap', 0) - non_interest_liab).clip(lower=1e-6)

    # ROIC
    df['calc_roic'] = np.where(
        invested_capital > 0,
        (nopat / invested_capital) * 100,
        np.nan
    )
    df['calc_roce'] = df['calc_roic']    

    # 7. Cash flow ratios
    df['calc_ocf_to_debt'] = np.where(
        df['total_liab'] > 0, df['n_cashflow_act'] / df['total_liab'], np.nan
    )
    df['calc_cf_sales'] = np.where(
        df[revenue_field] > 0, df['n_cashflow_act'] / df[revenue_field], np.nan
    )
    
    # 8. Quarterly indicators - calculate for all periods
    # In Tushare, q_ metrics may be available for both quarterly and annual reports
    df['calc_q_netprofit_margin'] = df['calc_netprofit_margin']  # Same as annual for now
    df['calc_q_roe'] = df['calc_roe_waa']  # Same as annual ROE for now
    
    # 9. Growth (YoY) - improved calculation with better data handling
    # Ensure proper sorting for YoY calculations
    df = df.sort_values(['ts_code', 'report_period'])

    # Determine if this is quarterly data by checking report periods
    quarterly_endings = ['0331', '0630', '0930', '1231']
    df['is_quarterly'] = df['report_period'].astype(str).str.endswith(tuple(quarterly_endings))

    # For each stock, determine if it has quarterly data
    stock_has_quarterly = df.groupby('ts_code')['is_quarterly'].any()

    # Calculate YoY changes based on data type per stock
    def calculate_yoy(series, stock_code):
        if stock_has_quarterly.get(stock_code, False):
            # Quarterly data: compare with 4 periods ago
            return series.pct_change(periods=4, fill_method=None) * 100
        else:
            # Annual data: compare with 1 period ago
            return series.pct_change(periods=1, fill_method=None) * 100

    # Apply YoY calculation per stock
    df['calc_netprofit_yoy'] = df.groupby('ts_code')['n_income_attr_p'].transform(
        lambda x: calculate_yoy(x, x.name)
    )
    df['calc_op_yoy'] = df.groupby('ts_code')['operate_profit'].transform(
        lambda x: calculate_yoy(x, x.name)
    )

    # Clean up temporary column
    df = df.drop('is_quarterly', axis=1)
    
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
    
    # 12. Per-share indicators (extended)
    # Book Value Per Share (BVPS/BPS) - 每股净资产
    equity_field = 'total_hldr_eqy_exc_min_int' if 'total_hldr_eqy_exc_min_int' in df.columns else 'total_hldr_eqy_inc_min_int'
    df['calc_bvps'] = np.where(
        df['total_share'] > 0,
        df[equity_field] / df['total_share'],
        np.nan
    )
    # Round to match API precision (typically 4 decimal places for BPS)
    df['calc_bvps'] = np.round(df['calc_bvps'], 4)
    df['calc_bps'] = df['calc_bvps']  # BPS is same as BVPS

    # Earnings Per Share (EPS) - 基本每股收益 (if not already calculated)
    if 'calc_eps' not in df.columns:
        df['calc_eps'] = df['basic_eps']  # Use from API if available
        df['calc_dt_eps'] = df['dt_eps']  # Diluted EPS

    # Operating Cash Flow Per Share (OCFPS) - 每股经营现金流
    df['calc_ocfps'] = np.where(
        df['total_share'] > 0,
        df['n_cashflow_act'] / df['total_share'],
        np.nan
    )

    # Cash Flow Per Share (CFPS) - 每股现金流净额
    df['calc_cfps'] = np.where(
        df['total_share'] > 0,
        df['n_incr_cash_cash_equ'] / df['total_share'],
        np.nan
    )

    # Revenue Per Share - 每股营业收入
    df['calc_revenue_ps'] = np.where(
        df['total_share'] > 0,
        df['total_revenue'] / df['total_share'],
        np.nan
    )

    # 13. Additional turnover ratios (extended)
    # Current Asset Turnover - 流动资产周转率
    df['calc_currentasset_turn'] = np.where(
        df['avg_total_cur_assets'] > 0,
        df['revenue'] / df['avg_total_cur_assets'],
        np.nan
    )

    # Fixed Assets Turnover - 固定资产周转率
    df['calc_fix_assets_turn'] = np.where(
        df['avg_fix_assets'] > 0,
        df['revenue'] / df['avg_fix_assets'],
        np.nan
    )

    # 14. Additional structure ratios (extended)
    # Non-current Assets to Total Assets - 非流动资产/总资产
    non_current_assets = df['total_assets'] - df['total_cur_assets']
    df['calc_nca_to_assets'] = np.where(
        df['total_assets'] > 0,
        non_current_assets / df['total_assets'] * 100,
        np.nan
    )

    # Non-current Liabilities to Total Liabilities - 非流动负债/负债合计
    non_current_liab = df['total_liab'] - df['total_cur_liab']
    df['calc_ncl_to_liab'] = np.where(
        df['total_liab'] > 0,
        non_current_liab / df['total_liab'] * 100,
        np.nan
    )

    # Operating Profit to Total Profit - 营业利润/利润总额
    df['calc_op_to_tp'] = np.where(
        df['total_profit'] != 0,
        df['operate_profit'] / df['total_profit'] * 100,
        np.nan
    )

    # Tax to Total Profit - 税项/利润总额
    df['calc_tax_to_tp'] = np.where(
        df['total_profit'] != 0,
        df['income_tax'] / df['total_profit'] * 100,
        np.nan
    )

    # Operating Profit to Current Liabilities - 营业利润/流动负债
    df['calc_op_to_cl'] = np.where(
        df['total_cur_liab'] > 0,
        df['operate_profit'] / df['total_cur_liab'],
        np.nan
    )

    # Operating Cash Flow to Current Liabilities - 经营现金流/流动负债
    df['calc_ocf_to_cl'] = np.where(
        df['total_cur_liab'] > 0,
        df['n_cashflow_act'] / df['total_cur_liab'],
        np.nan
    )

    # Equity to Total Liabilities - 归母权益/负债合计
    df['calc_eqt_to_liab'] = np.where(
        df['total_liab'] > 0,
        df['total_hldr_eqy_inc_min_int'] / df['total_liab'] * 100,
        np.nan
    )

    for field in df.columns:
        if field.startswith('calc_'):
            field_null_count = df[field].isna().sum()
            logger.debug(f"{field}_null_pct: {field_null_count / len(df) * 100.0}")

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
        # Basic margins and ratios
        'grossprofit_margin': 'calc_grossprofit_margin',
        'netprofit_margin': 'calc_netprofit_margin',
        'ebitda_margin': 'calc_ebitda_margin',

        # Per-share indicators
        'bvps': 'calc_bvps',
        'bps': 'calc_bps',
        'ocfps': 'calc_ocfps',
        'cfps': 'calc_cfps',
        'revenue_ps': 'calc_revenue_ps',

        # Solvency ratios
        'current_ratio': 'calc_current_ratio',
        'quick_ratio': 'calc_quick_ratio',
        'cash_ratio': 'calc_cash_ratio',
        'debt_to_assets': 'calc_debt_to_assets',
        'assets_to_eqt': 'calc_assets_to_eqt',
        'ebit_to_interest': 'calc_ebit_to_interest',

        # Turnover ratios
        'inv_turn': 'calc_inv_turn',
        'ar_turn': 'calc_ar_turn',
        'assets_turn': 'calc_assets_turn',
        'currentasset_turn': 'calc_currentasset_turn',
        'fix_assets_turn': 'calc_fix_assets_turn',

        # Profitability ratios
        'roe_waa': 'calc_roe_waa',
        'roa': 'calc_roa',
        'npta': 'calc_npta',
        'roic': 'calc_roic',

        # Cash flow ratios
        'ocf_to_debt': 'calc_ocf_to_debt',
        'cf_sales': 'calc_cf_sales',

        # Quarterly ratios
        'q_netprofit_margin': 'calc_q_netprofit_margin',
        'q_roe': 'calc_q_roe',

        # Growth ratios
        'netprofit_yoy': 'calc_netprofit_yoy',
        'op_yoy': 'calc_op_yoy',

        # Cost and expense ratios
        'cogs_of_sales': 'calc_cogs_of_sales',
        'expense_of_sales': 'calc_expense_of_sales',

        # Asset structure ratios
        'ca_to_assets': 'calc_ca_to_assets',
        'nca_to_assets': 'calc_nca_to_assets',
        'currentdebt_to_debt': 'calc_currentdebt_to_debt',

        # Additional structure ratios
        'ncl_to_liab': 'calc_ncl_to_liab',
        'op_to_tp': 'calc_op_to_tp',
        'tax_to_tp': 'calc_tax_to_tp',
        'op_to_cl': 'calc_op_to_cl',
        'ocf_to_cl': 'calc_ocf_to_cl',
        'eqt_to_liab': 'calc_eqt_to_liab'
    }
    
    consistency_summary = {}
    for api_col, calc_col in diff_map.items():
        logger.debug(f"Comparing {api_col} and {calc_col}")
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
    computed_df = compute_basic_indicators(income_df, balance_df, cashflow_df, fina_df)
    
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
        sample_cols = ['ts_code', 'report_period', 'grossprofit_margin_abs_diff', 'netprofit_margin_abs_diff',
                      'bvps_abs_diff', 'ocfps_abs_diff', 'currentasset_turn_abs_diff', 'fix_assets_turn_abs_diff',
                      'nca_to_assets_abs_diff', 'op_to_tp_abs_diff', 'eqt_to_liab_abs_diff']  # Extended sample
        available_cols = [col for col in sample_cols if col in validation_df.columns]
        if available_cols:
            print(validation_df[available_cols].head().to_string(index=False))
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