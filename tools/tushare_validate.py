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
    'profit_to_op', 'basic_eps_yoy', 'dt_eps_yoy', 'cfps_yoy', 
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

def get_quarterly_value(data_dict, key, period):
    """
    Extract quarterly value from cumulative data.
    This function calculates the actual quarterly value by subtracting previous cumulative values.
    """
    if period.endswith('0331'):  # Q1 is already quarterly
        return data_dict.get(period, {}).get(key, 0)
    elif period.endswith('0630'):  # H1 - Q1
        current = data_dict.get(period, {}).get(key, 0)
        q1_period = period[:4] + '0331'
        q1_value = data_dict.get(q1_period, {}).get(key, 0)
        return current - q1_value
    elif period.endswith('0930'):  # Q1-Q3 - H1
        current = data_dict.get(period, {}).get(key, 0)
        h1_period = period[:4] + '0630'
        h1_value = data_dict.get(h1_period, {}).get(key, 0)
        return current - h1_value
    elif period.endswith('1231'):  # Full year - Q1-Q3
        current = data_dict.get(period, {}).get(key, 0)
        q3_period = period[:4] + '0930'
        q3_value = data_dict.get(q3_period, {}).get(key, 0)
        return current - q3_value
    return 0


def check_consecutive_quarters(periods, target_period):
    """
    Check if we have 4 consecutive quarters ending with target_period.
    Returns the 4 consecutive quarters if available, otherwise empty list.
    """
    if not periods:
        return []

    # Convert all periods to quarter tuples for easier processing
    def period_to_quarter(period):
        year = int(period[:4])
        month_day = period[4:]
        if month_day == '0331':
            return (year, 1)
        elif month_day == '0630':
            return (year, 2)
        elif month_day == '0930':
            return (year, 3)
        elif month_day == '1231':
            return (year, 4)
        return None

    # Get target quarter
    target_quarter = period_to_quarter(target_period)
    if not target_quarter:
        return []

    # Convert all available periods to quarters and sort chronologically
    available_quarters = []
    period_to_quarter_map = {}

    for p in periods:
        q = period_to_quarter(p)
        if q:
            available_quarters.append(q)
            period_to_quarter_map[q] = p

    available_quarters.sort()  # Sort chronologically (earliest first)

    # Find target quarter in available quarters
    if target_quarter not in available_quarters:
        return []

    target_idx = available_quarters.index(target_quarter)

    # Need 4 consecutive quarters ending with target quarter
    # So we need quarters: target_idx-3, target_idx-2, target_idx-1, target_idx
    if target_idx < 3:
        return []  # Not enough previous quarters

    candidate_quarters = available_quarters[target_idx-3:target_idx+1]

    # Check if these 4 quarters are consecutive
    for i in range(3):
        curr_year, curr_q = candidate_quarters[i]
        next_year, next_q = candidate_quarters[i + 1]

        # Check consecutive quarters
        if not ((next_year == curr_year and next_q == curr_q + 1) or
                (next_year == curr_year + 1 and curr_q == 4 and next_q == 1)):
            return []

    # Return the corresponding periods in reverse chronological order (most recent first)
    result_periods = [period_to_quarter_map[q] for q in candidate_quarters[::-1]]
    return result_periods


def calculate_ttm_indicators(stock_df, report_period):
    """
    Calculate TTM (Trailing Twelve Months) indicators for a given report period.
    stock_df should contain data for a single stock.
    Only calculates if we have complete consecutive quarterly data for the past 4 quarters.
    Returns a dictionary of TTM indicators, or empty dict if data is incomplete.
    """
    if stock_df.empty:
        return {}

    # stock_df is already filtered for a single stock
    stock_data = stock_df.sort_values('report_period')

    # Convert to data dict for easier access
    data_dict = {}
    for _, row in stock_data.iterrows():
        period = row['report_period']
        data_dict[period] = {
            'n_income_attr_p': row.get('n_income_attr_p', 0),
            'total_revenue': row.get('total_revenue', 0),
            'revenue': row.get('revenue', row.get('total_revenue', 0)),
            'im_net_cashflow_oper_act': row.get('im_net_cashflow_oper_act', row.get('n_cashflow_act', 0)),
            'n_cashflow_act': row.get('n_cashflow_act', 0),
            'total_cogs': row.get('total_cogs', 0),
            'oper_cost': row.get('oper_cost', 0),
            'total_hldr_eqy_exc_min_int': row.get('total_hldr_eqy_exc_min_int', 0),
            'total_assets': row.get('total_assets', 0),
            'total_share': row.get('total_share', 0)
        }

    # Get all available periods up to report_period
    available_periods = [p for p in data_dict.keys() if p <= report_period]

    # Check for consecutive quarters
    ttm_periods = check_consecutive_quarters(available_periods, report_period)

    if len(ttm_periods) != 4:
        # Not enough consecutive quarterly data for TTM calculation
        return {}

    # Calculate quarterly values for the TTM periods
    q_net = {}
    q_rev = {}
    q_ocf = {}
    q_cogs = {}
    q_oper_cost = {}

    for p in ttm_periods:
        q_net[p] = get_quarterly_value(data_dict, 'n_income_attr_p', p)
        q_rev[p] = get_quarterly_value(data_dict, 'total_revenue', p)
        q_ocf[p] = get_quarterly_value(data_dict, 'im_net_cashflow_oper_act', p)
        q_cogs[p] = get_quarterly_value(data_dict, 'total_cogs', p)
        q_oper_cost[p] = get_quarterly_value(data_dict, 'oper_cost', p)

    # Calculate TTM totals
    ttm_net = sum(q_net.get(p, 0) for p in ttm_periods)
    ttm_rev = sum(q_rev.get(p, 0) for p in ttm_periods)
    ttm_ocf = sum(q_ocf.get(p, 0) for p in ttm_periods)
    ttm_cogs = sum(q_cogs.get(p, 0) for p in ttm_periods)
    ttm_oper_cost = sum(q_oper_cost.get(p, 0) for p in ttm_periods)
    ttm_gross = ttm_rev - ttm_oper_cost
    logger.debug(f"report_period: {report_period}, ttm_rev: {ttm_rev}, ttm_cogs: {ttm_cogs}")

    # Get shares (use the most recent period's shares)
    shares = data_dict.get(report_period, {}).get('total_share', 0)
    if shares == 0:
        # Fallback to other periods
        for p in ttm_periods:
            shares = data_dict.get(p, {}).get('total_share', 0)
            if shares > 0:
                break

    if shares == 0:
        return {}

    # Calculate per-share TTM indicators
    eps_ttm = ttm_net / shares if shares > 0 else 0
    revenue_ps_ttm = ttm_rev / shares if shares > 0 else 0
    ocfps_ttm = ttm_ocf / shares if shares > 0 else 0
    cfps_ttm = ttm_ocf / shares if shares > 0 else 0  # Using OCF as CFPS

    # Calculate ROE_TTM and ROA_TTM
    # Use average equity/assets over the TTM period
    equity_values = []
    asset_values = []

    for p in ttm_periods:
        eq = data_dict.get(p, {}).get('total_hldr_eqy_exc_min_int', 0)
        assets = data_dict.get(p, {}).get('total_assets', 0)
        if eq > 0:
            equity_values.append(eq)
        if assets > 0:
            asset_values.append(assets)

    roe_ttm = (ttm_net / equity_values[0] * 100) if equity_values and equity_values[0] > 0 else 0
    roa_ttm = (ttm_net / asset_values[0] * 100) if asset_values and asset_values[0] > 0 else 0
    # Calculate margin ratios
    netprofit_margin_ttm = (ttm_net / ttm_rev * 100) if ttm_rev > 0 else 0
    logger.debug(f"ttm_gross: {ttm_gross}, ttm_rev: {ttm_rev}")
    grossprofit_margin_ttm = (ttm_gross / ttm_rev * 100) if ttm_rev > 0 else 0

    # Calculate 3-year CAGR (36 months) for revenue and net income
    # Compare current period with the same period 3 years ago
    revenue_cagr_3y = None
    ni_cagr_3y = None

    # Calculate the period 3 years ago (same quarter)
    def get_period_3_years_ago(period):
        """Get the same period 3 years ago"""
        try:
            year = int(period[:4]) - 3
            month_day = period[4:]
            return f"{year}{month_day}"
        except (ValueError, IndexError):
            return None

    start_period = get_period_3_years_ago(report_period)

    if start_period and start_period in data_dict:
        # Get values for current period and 3 years ago
        start_revenue = data_dict.get(start_period, {}).get('total_revenue', 0)
        end_revenue = data_dict.get(report_period, {}).get('total_revenue', 0)

        start_ni = data_dict.get(start_period, {}).get('n_income_attr_p', 0)
        end_ni = data_dict.get(report_period, {}).get('n_income_attr_p', 0)

        logger.debug(f"CAGR calculation: {start_period} -> {report_period}")
        logger.debug(f"Revenue: {start_revenue} -> {end_revenue}")
        logger.debug(f"NI: {start_ni} -> {end_ni}")

        # Calculate 3-year CAGR for revenue
        if start_revenue > 0 and end_revenue > 0:
            try:
                cagr_ratio = (end_revenue / start_revenue) ** (1/3) - 1
                # Check if result is complex (shouldn't happen with positive inputs, but safety check)
                if isinstance(cagr_ratio, complex):
                    revenue_cagr_3y = None
                else:
                    revenue_cagr_3y = round(float(cagr_ratio) * 100, 4)  # Convert to percentage
            except (ValueError, OverflowError, ZeroDivisionError):
                revenue_cagr_3y = None

        # Calculate 3-year CAGR for net income
        if start_ni > 0 and end_ni > 0:  # Both must be positive for meaningful CAGR
            try:
                cagr_ratio = (end_ni / start_ni) ** (1/3) - 1
                # Check if result is complex (shouldn't happen with positive inputs, but safety check)
                if isinstance(cagr_ratio, complex):
                    ni_cagr_3y = None
                else:
                    ni_cagr_3y = round(float(cagr_ratio) * 100, 4)  # Convert to percentage
            except (ValueError, OverflowError, ZeroDivisionError):
                ni_cagr_3y = None
        else:
            # If either value is not positive, CAGR calculation doesn't make sense
            ni_cagr_3y = None

    return {
        'eps_ttm': round(eps_ttm, 4),
        'revenue_ps_ttm': round(revenue_ps_ttm, 4),
        'ocfps_ttm': round(ocfps_ttm, 4),
        'cfps_ttm': round(cfps_ttm, 4),
        'roe_ttm': round(roe_ttm, 4),
        'roa_ttm': round(roa_ttm, 4),
        'netprofit_margin_ttm': round(netprofit_margin_ttm, 4),
        'grossprofit_margin_ttm': round(grossprofit_margin_ttm, 4),
        'revenue_cagr_3y': revenue_cagr_3y,
        'netincome_cagr_3y': ni_cagr_3y
    }


def compute_basic_indicators(income_df, balance_df, cashflow_df, fina_df, stocks):
    """Compute extended basic indicators from primitives"""
    if income_df.empty or balance_df.empty or cashflow_df.empty:
        return pd.DataFrame()

    # Filter by stocks if specified
    if stocks is not None:
        stock_list = [s.strip() for s in stocks.split(',') if s.strip()]
        if stock_list:
            logger.info(f"Filtering data for stocks: {stock_list}")
            if not income_df.empty and 'ts_code' in income_df.columns:
                income_df = income_df[income_df['ts_code'].isin(stock_list)].copy()
            if not balance_df.empty and 'ts_code' in balance_df.columns:
                balance_df = balance_df[balance_df['ts_code'].isin(stock_list)].copy()
            if not cashflow_df.empty and 'ts_code' in cashflow_df.columns:
                cashflow_df = cashflow_df[cashflow_df['ts_code'].isin(stock_list)].copy()
            if not fina_df.empty and 'ts_code' in fina_df.columns:
                fina_df = fina_df[fina_df['ts_code'].isin(stock_list)].copy()

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

    # Calculate TTM indicators for each stock and period
    print("Calculating TTM (Trailing Twelve Months) indicators...")
    ttm_indicators = []

    # Group by stock to calculate TTM for each stock
    for stock_code, stock_df in df.groupby('ts_code'):
        stock_df = stock_df.sort_values('report_period')

        for _, row in stock_df.iterrows():
            report_period = row['report_period']
            ttm_result = calculate_ttm_indicators(stock_df, report_period)

            if ttm_result:
                ttm_row = {
                    'ts_code': stock_code,
                    'report_period': report_period,
                    **ttm_result
                }
                ttm_indicators.append(ttm_row)

    # Convert TTM indicators to DataFrame and merge with main df
    if ttm_indicators:
        ttm_df = pd.DataFrame(ttm_indicators)
        df = df.merge(ttm_df, on=['ts_code', 'report_period'], how='left')
        print(f"Added TTM indicators for {len(ttm_df)} stock-period combinations")
    else:
        print("No TTM indicators calculated (insufficient quarterly data)")

    df['equity_growth_yoy'] = df['equity_yoy'].fillna(0)
    df['oper_rev_yoy'] = df['or_yoy'].fillna(0)
    df['debt_to_assets'] = df['debt_to_assets'].fillna(0)
    df['debt_to_equity'] = df['debt_to_eqt'].fillna(0)
    df['current_ratio'] = df['current_ratio'].fillna(0)
    df['quick_ratio'] = df['quick_ratio'].fillna(0)
    df['cash_ratio'] = df['cash_ratio'].fillna(0)
    df['ca_turn'] = df['ca_turn'].fillna(0)
    df.to_csv('computed_indicators.csv', index=False)
    return df

def deduplicate_dataframes(income_df, balance_df, cashflow_df, fina_df):
    """
    Remove duplicates from dataframes based on primary key (ts_code, report_period).
    This ensures that for each stock and period combination, only the last (most recent/corrected) record is kept.
    This follows the same pattern as update_a_stock_financial_profile.py
    """
    # Add report_period column to all dataframes if not present
    dataframes = [
        ("income_df", income_df),
        ("balance_df", balance_df),
        ("cashflow_df", cashflow_df),
        ("fina_df", fina_df)
    ]

    for name, df in dataframes:
        if not df.empty and len(df) > 0:
            # Add report_period column if not present
            if 'report_period' not in df.columns and 'end_date' in df.columns:
                df = df.copy()
                df['report_period'] = df['end_date'].str.replace('-', '')
                if name == "income_df":
                    income_df = df
                elif name == "balance_df":
                    balance_df = df
                elif name == "cashflow_df":
                    cashflow_df = df
                elif name == "fina_df":
                    fina_df = df

    # Now deduplicate based on (ts_code, report_period)
    for name, df in [("income_df", income_df), ("balance_df", balance_df), ("cashflow_df", cashflow_df), ("fina_df", fina_df)]:
        if not df.empty and len(df) > 0 and 'ts_code' in df.columns and 'report_period' in df.columns:
            initial_count = len(df)

            # Debug: Check for duplicates before removal
            duplicate_check = df.groupby(['ts_code', 'report_period']).size()
            duplicates_found = duplicate_check[duplicate_check > 1]

            if len(duplicates_found) > 0:
                print(f"Found {len(duplicates_found)} duplicate groups in {name} before removal:")
                for (ts_code, report_period), count in duplicates_found.items():
                    print(f"  {ts_code} {report_period}: {count} duplicates")

                # Remove duplicates, keeping the last record (potentially more updated/corrected data)
                df = df.drop_duplicates(subset=['ts_code', 'report_period'], keep='last')
                final_count = len(df)

                if final_count < initial_count:
                    print(f"Removed {initial_count - final_count} duplicates from {name}, kept {final_count} unique records (latest)")

                # Update the dataframe variable
                if name == "income_df":
                    income_df = df
                elif name == "balance_df":
                    balance_df = df
                elif name == "cashflow_df":
                    cashflow_df = df
                elif name == "fina_df":
                    fina_df = df

    return income_df, balance_df, cashflow_df, fina_df


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
        'roe': 'calc_roe',
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

    # Deduplicate dataframes to ensure data quality
    print("1.5. Deduplicating data...")
    logger.info(f"income_df: {len(income_df)}, balance_df: {len(balance_df)}, cashflow_df: {len(cashflow_df)}, fina_df: {len(fina_df)}")
    income_df, balance_df, cashflow_df, fina_df = deduplicate_dataframes(income_df, balance_df, cashflow_df, fina_df)
    logger.info(f"after deduplicate, income_df: {len(income_df)}, balance_df: {len(balance_df)}, cashflow_df: {len(cashflow_df)}, fina_df: {len(fina_df)}")

    # Compute (extended)
    print("2. Computing extended indicators...")
    computed_df = compute_basic_indicators(income_df, balance_df, cashflow_df, fina_df, stocks)
    
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
    parser.add_argument("--end_date", type=str, help="End date in YYYY-MM-DD or YYYYMMDD format.", default=datetime.now().strftime('%Y-12-31'))
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