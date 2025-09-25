import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
import argparse
import time
import logging

from util import (
    setup_logging, CacheManager, init_tushare, retry_api_call
)

setup_logging(log_file='tushare_validate.log')
logger = logging.getLogger(__name__)

cache_manager = CacheManager()

tushare_pro = init_tushare()

# API field name list (all three major financial statements contain these base fields)
# Note: API returns 'end_date' but database stores as 'ann_date'
API_COMMON_FIELDS = ['ts_code', 'ann_date', 'end_date', 'report_type']

# Financial indicators base fields (does not include report_type)
INDICATOR_BASE_FIELDS = ['ts_code', 'ann_date', 'end_date']  # Keep end_date for API call, will be mapped later

# Income statement fields (core primitives for calculations)
INCOME_COLUMNS = [
    'basic_eps', 'diluted_eps', 'total_revenue', 'revenue',
    'total_cogs', 'oper_cost', 'sell_exp', 'admin_exp', 'fin_exp',
    'assets_impair_loss', 'operate_profit', 'non_oper_income', 'non_oper_exp',
    'total_profit', 'income_tax', 'n_income', 'n_income_attr_p', 'ebit',
    'ebitda', 'invest_income'
]

# Balance sheet fields (core primitives)
BALANCE_COLUMNS = [
    'total_share', 'cap_rese', 'undistr_porfit', 'surplus_rese', 'money_cap',
    'accounts_receiv', 'oth_receiv', 'prepayment', 'inventories',
    'oth_cur_assets', 'total_cur_assets', 'htm_invest', 'fix_assets',
    'intan_assets', 'defer_tax_assets', 'total_nca', 'total_assets',
    'acct_payable', 'payroll_payable', 'taxes_payable', 'oth_payable',
    'total_cur_liab', 'defer_inc_non_cur_liab', 'total_ncl', 'total_liab',
    'total_hldr_eqy_exc_min_int', 'total_hldr_eqy_inc_min_int',
    'total_liab_hldr_eqy', 'oth_pay_total', 'accounts_receiv_bill',
    'accounts_pay', 'oth_rcv_total', 'fix_assets_total'
]

# Cash flow statement fields (core primitives)
CASHFLOW_COLUMNS = [
    'net_profit', 'finan_exp', 'c_fr_sale_sg', 'c_inf_fr_operate_a',
    'c_paid_goods_s', 'c_paid_to_for_empl', 'c_paid_for_taxes',
    'n_cashflow_act', 'n_cashflow_inv_act', 'free_cashflow',
    'n_cash_flows_fnc_act', 'n_incr_cash_cash_equ', 'c_cash_equ_beg_period',
    'c_cash_equ_end_period', 'im_net_cashflow_oper_act', 'end_bal_cash',
    'beg_bal_cash'
]

# Financial indicators fields (calculated metrics)
INDICATOR_COLUMNS = [
    'eps', 'dt_eps', 'revenue_ps', 'bps', 'cfps', 'gross_margin',
    'netprofit_margin', 'grossprofit_margin', 'current_ratio', 'quick_ratio',
    'cash_ratio', 'inv_turn', 'ar_turn', 'ca_turn', 'fa_turn', 'assets_turn',
    'debt_to_assets', 'debt_to_eqt', 'roe', 'roa', 'roic', 'netprofit_yoy',
    'or_yoy', 'basic_eps_yoy', 'assets_yoy', 'eqt_yoy', 'ocf_yoy', 'roe_yoy',
    'equity_yoy', 'rd_exp', 'fcff_ps'
]

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
            ts_codes_str = ','.join(stocks_list)
        else:
            stocks_list = []
            ts_codes_str = ''

        income_dfs = []
        balance_dfs = []
        cashflow_dfs = []
        fina_dfs = []

        for period in periods:
            try:
                # Income statement (doc_id=33) - focus on basics
                income_df = call_tushare_api_with_retry(
                    tushare_pro.income_vip,
                    ts_code=ts_codes_str,
                    period=period,
                    fields=','.join(API_COMMON_FIELDS + INCOME_COLUMNS)
                )
                if not income_df.empty:
                    income_dfs.append(income_df)

                # Balance sheet (doc_id=36) - for equity/assets
                balance_df = call_tushare_api_with_retry(
                    tushare_pro.balancesheet_vip,
                    ts_code=ts_codes_str,
                    period=period,
                    fields=','.join(API_COMMON_FIELDS + BALANCE_COLUMNS)
                )
                if not balance_df.empty:
                    balance_dfs.append(balance_df)

                # Cash flow (doc_id=44) - basic for completeness
                cashflow_df = call_tushare_api_with_retry(
                    tushare_pro.cashflow_vip,
                    ts_code=ts_codes_str,
                    period=period,
                    fields=','.join(API_COMMON_FIELDS + CASHFLOW_COLUMNS)
                )
                if not cashflow_df.empty:
                    cashflow_dfs.append(cashflow_df)

                # Financial indicators (doc_id=79) - for validation (extended fields)
                indicator_fields = INDICATOR_BASE_FIELDS + INDICATOR_COLUMNS
                fina_df = call_tushare_api_with_retry(
                    tushare_pro.fina_indicator_vip,
                    ts_code=ts_codes_str,
                    period=period,
                    fields=','.join(indicator_fields)
                )
                if not fina_df.empty:
                    fina_dfs.append(fina_df)
            except Exception as e:
                logger.error(f"Error fetching data for period {period} after retries: {e}")
                continue

        # Concatenate all dataframes
        income_df = pd.concat(income_dfs, ignore_index=True) if income_dfs else pd.DataFrame()
        balance_df = pd.concat(balance_dfs, ignore_index=True) if balance_dfs else pd.DataFrame()
        cashflow_df = pd.concat(cashflow_dfs, ignore_index=True) if cashflow_dfs else pd.DataFrame()
        fina_df = pd.concat(fina_dfs, ignore_index=True) if fina_dfs else pd.DataFrame()

        return income_df, balance_df, cashflow_df, fina_df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def fetch_tushare_data(stocks: str, periods: List[str]):
    """Fetch data from multiple Tushare interfaces with caching support"""
    # Try to load from cache first
    income_df = cache_manager.load_from_cache('income_data')
    balance_df = cache_manager.load_from_cache('balance_data')
    cashflow_df = cache_manager.load_from_cache('cashflow_data')
    fina_df = cache_manager.load_from_cache('fina_data')

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
            cache_manager.save_to_cache("income_data", api_income_df)
            if income_df.empty:
                income_df = api_income_df

        if not api_balance_df.empty:
            cache_manager.save_to_cache("balance_data", api_balance_df)
            if balance_df.empty:
                balance_df = api_balance_df

        if not api_cashflow_df.empty:
            cache_manager.save_to_cache("cashflow_data", api_cashflow_df)
            if cashflow_df.empty:
                cashflow_df = api_cashflow_df

        if not api_fina_df.empty:
            cache_manager.save_to_cache("fina_data", api_fina_df)
            if fina_df.empty:
                fina_df = api_fina_df

        return income_df, balance_df, cashflow_df, fina_df

def calculate_quarterly_values(group, columns):
    """Calculate quarterly values using vectorized operations within each year"""
    group = group.sort_values('report_period')
    group['year'] = group['report_period'].str[:4]
    for col in columns:
        group['q_' + col] = group.groupby('year')[col].diff().fillna(group[col])
    return group.drop(columns=['year'])

def calculate_ttm_indicators(df):
    """
    Vectorized calculation of TTM indicators.
    Assumes df has 'ts_code', 'report_period', and required columns.
    Returns df with added TTM columns.
    """
    if df.empty:
        return df

    # 转换为datetime以便处理时间序列
    df['report_date'] = pd.to_datetime(df['report_period'], format='%Y%m%d')

    # 为每个ts_code补全中间缺失的季度序列
    def complete_quarters(group):
        # 找到实际存在数据的日期范围
        existing_dates = group['report_date'].dropna().sort_values()

        if len(existing_dates) < 2:
            # 如果数据点太少，无法确定补全范围，直接返回原数据
            return group

        min_date = existing_dates.min()
        max_date = existing_dates.max()

        # 生成从最早数据到最晚数据的完整季度末序列
        full_dates = pd.date_range(start=min_date, end=max_date, freq='QE-SEP')
        full_df = pd.DataFrame({'report_date': full_dates})
        full_df['report_period'] = full_df['report_date'].dt.strftime('%Y%m%d')
        full_df['ts_code'] = group['ts_code'].iloc[0]  # 添加ts_code

        # 左合并原数据，缺失处NA
        merged = pd.merge(full_df, group, on=['ts_code', 'report_period', 'report_date'], how='left')

        # 标记补全的行
        merged['missing'] = merged['n_income_attr_p'].isna().astype(int)

        # 只记录实际缺失的中间数据（不是两头的）
        missing_periods = merged[merged['n_income_attr_p'].isna()]['report_period'].tolist()

        # 过滤掉两头的缺失数据：如果某个时期在现有数据的最小日期之前或最大日期之后，则不算作缺失
        existing_periods = set(group['report_period'].dropna())
        if existing_periods:
            min_existing_period = min(existing_periods)
            max_existing_period = max(existing_periods)

            # 只保留在现有数据范围内的缺失时期
            filtered_missing = [
                period for period in missing_periods
                if min_existing_period <= period <= max_existing_period
            ]

            if filtered_missing:
                logger.warning(f"Inserted NA for missing intermediate periods in {group['ts_code'].iloc[0]}: {filtered_missing}")

        return merged
    df = df.groupby('ts_code').apply(complete_quarters).reset_index(drop=True)
    # 填充NA：flow填0，stock ffill/bfill
    flow_cols = ['n_income_attr_p', 'total_revenue', 'im_net_cashflow_oper_act', 'total_cogs', 'oper_cost']
    stock_cols = ['total_hldr_eqy_exc_min_int', 'total_assets', 'total_share']
    df[flow_cols] = df[flow_cols].fillna(0)
    df[stock_cols] = df.groupby('ts_code')[stock_cols].ffill().bfill()

    quarterly_columns = ['n_income_attr_p', 'total_revenue', 'im_net_cashflow_oper_act', 
                         'total_cogs', 'oper_cost']
    df = df.groupby('ts_code').apply(lambda g: calculate_quarterly_values(g, quarterly_columns)).reset_index(drop=True)

    # Sort by ts_code and report_period
    df = df.sort_values(['ts_code', 'report_period'])

    # Calculate rolling TTM sums for quarterly values
    ttm_columns = {col: 'ttm_' + col for col in quarterly_columns}
    for q_col, ttm_col in ttm_columns.items():
        df[ttm_col] = df.groupby('ts_code')['q_' + q_col].rolling(window=4, min_periods=4).sum().reset_index(level=0, drop=True)

    # Drop rows where TTM is NaN (insufficient history)
    #df = df.dropna(subset=list(ttm_columns.values()))

    # Calculate TTM gross
    df['ttm_gross'] = df['ttm_total_revenue'] - df['ttm_oper_cost']

    # Per-share calculations (vectorized)
    df['eps_ttm'] = np.where(df['total_share'] > 0, df['ttm_n_income_attr_p'] / df['total_share'], 0)
    df['revenue_ps_ttm'] = np.where(df['total_share'] > 0, df['ttm_total_revenue'] / df['total_share'], 0)
    df['ocfps_ttm'] = np.where(df['total_share'] > 0, df['ttm_im_net_cashflow_oper_act'] / df['total_share'], 0)
    df['cfps_ttm'] = df['ocfps_ttm']  # Assuming CFPS uses OCF

    # ROE and ROA (using period-end values)
    df['roe_ttm'] = np.where(df['total_hldr_eqy_exc_min_int'] > 0, 
                             (df['ttm_n_income_attr_p'] / df['total_hldr_eqy_exc_min_int']) * 100, 0)
    df['roa_ttm'] = np.where(df['total_assets'] > 0, 
                             (df['ttm_n_income_attr_p'] / df['total_assets']) * 100, 0)

    # Margins
    df['netprofit_margin_ttm'] = np.where(df['ttm_total_revenue'] > 0, 
                                          (df['ttm_n_income_attr_p'] / df['ttm_total_revenue']) * 100, 0)
    df['grossprofit_margin_ttm'] = np.where(df['ttm_total_revenue'] > 0, 
                                            (df['ttm_gross'] / df['ttm_total_revenue']) * 100, 0)

    # CAGR (3-year, same quarter)
    df['revenue_3y_ago'] = df.groupby('ts_code')['total_revenue'].shift(12)
    df['ni_3y_ago'] = df.groupby('ts_code')['n_income_attr_p'].shift(12)
    
    mask_rev = (df['revenue_3y_ago'] > 0) & (df['total_revenue'] > 0)
    df['revenue_cagr_3y'] = np.where(mask_rev, 
                                     ((df['total_revenue'] / df['revenue_3y_ago']) ** (1/3) - 1) * 100, np.nan)
    
    mask_ni = (df['ni_3y_ago'] > 0) & (df['n_income_attr_p'] > 0)
    df['netincome_cagr_3y'] = np.where(mask_ni, 
                                       ((df['n_income_attr_p'] / df['ni_3y_ago']) ** (1/3) - 1) * 100, np.nan)

    # Round results
    round_cols = ['eps_ttm', 'revenue_ps_ttm', 'ocfps_ttm', 'cfps_ttm', 'roe_ttm', 'roa_ttm', 
                  'netprofit_margin_ttm', 'grossprofit_margin_ttm', 'revenue_cagr_3y', 'netincome_cagr_3y']
    df[round_cols] = df[round_cols].round(4)

    # Drop temporary columns
    drop_cols = list(ttm_columns.values()) + ['ttm_gross', 'revenue_3y_ago', 'ni_3y_ago'] + ['q_' + col for col in quarterly_columns]
    #df = df.drop(columns=drop_cols, errors='ignore')

    # Remove filled rows (missing=1) after calculations are complete
    if 'missing' in df.columns:
        original_count = len(df)
        df = df[df['missing'] != 1].copy()
        removed_count = original_count - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} filled rows after TTM/CAGR calculations")
        df = df.drop(columns=['missing'])

    return df

def compute_basic_indicators(income_df, balance_df, cashflow_df, fina_df, stocks):
    """Compute extended basic indicators from primitives"""
    if income_df.empty or balance_df.empty or cashflow_df.empty:
        return pd.DataFrame()

    # Filter by stocks if specified
    if stocks:
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
        logger.error(f"Merge failed due to missing key: {e}")
        logger.debug(f"Available columns in income_df: {list(income_df.columns)}")
        logger.debug(f"Available columns in balance_df: {list(balance_df.columns)}")
        logger.debug(f"Available columns in cashflow_df: {list(cashflow_df.columns)}")
        logger.debug(f"Available columns in fina_df: {list(fina_df.columns)}")
        return pd.DataFrame()
    df = df.copy()

    # Calculate TTM indicators vectorized
    logger.info("Calculating TTM (Trailing Twelve Months) indicators...")
    df = calculate_ttm_indicators(df)

    df['netprofit_yoy'] = df['netprofit_yoy'].fillna(0)
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

            # Remove duplicates using groupby and last
            df = df.groupby(['ts_code', 'report_period'], as_index=False).last()
            final_count = len(df)

            if final_count < initial_count:
                logger.info(f"Removed {initial_count - final_count} duplicates from {name}, kept {final_count} unique records (latest)")

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
        logger.warning("No common merge keys found between computed and fina dataframes")
        return pd.DataFrame(), {}

    try:
        merged = pd.merge(computed_df, fina_df, on=merge_keys, how='inner', suffixes=('_calc', '_api'))
    except KeyError as e:
        logger.error(f"Cross-validation merge failed: {e}")
        logger.debug(f"Available merge keys: {merge_keys}")
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
    logger.info(f"Starting Tushare cross-validation for {stocks} ({start_date} to {end_date})")

    # Fetch
    logger.info("1. Fetching data...")
    periods = generate_periods(start_date, end_date, period)
    income_df, balance_df, cashflow_df, fina_df = fetch_tushare_data(stocks, periods)

    # Deduplicate dataframes to ensure data quality
    logger.info("1. Deduplicating data...")
    logger.info(f"income_df: {len(income_df)}, balance_df: {len(balance_df)}, cashflow_df: {len(cashflow_df)}, fina_df: {len(fina_df)}")
    income_df, balance_df, cashflow_df, fina_df = deduplicate_dataframes(income_df, balance_df, cashflow_df, fina_df)
    logger.info(f"after deduplicate, income_df: {len(income_df)}, balance_df: {len(balance_df)}, cashflow_df: {len(cashflow_df)}, fina_df: {len(fina_df)}")

    # Compute (extended)
    logger.info("2. Computing extended indicators...")
    computed_df = compute_basic_indicators(income_df, balance_df, cashflow_df, fina_df, stocks)

    # Validate (extended)
    logger.info("3. Cross-validating consistency...")
    validation_df, consistency_summary = cross_validate_indicators(computed_df, fina_df)

    # Completeness
    logger.info("4. Checking completeness...")
    completeness = check_completeness([income_df, balance_df, cashflow_df, fina_df])
    
    # Report
    logger.info("=== Extended Consistency Summary (% Consistent) ===")
    for k, v in consistency_summary.items():
        status = "PASS" if v >= 95 else "WARN"
        logger.info(f"{k}: {v:.2f}% ({status})")

    logger.info("=== Completeness Report ===")
    for k, v in completeness.items():
        if isinstance(v, dict):
            logger.info(f"{k}: rows={v['rows']}, null_pct={v['null_pct']}%, coverage={v['coverage']}%")
        else:
            logger.info(f"{k}: {v}")

    logger.info("=== Detailed Validation (Sample) ===")
    if not validation_df.empty:
        sample_cols = ['ts_code', 'report_period', 'grossprofit_margin_abs_diff', 'netprofit_margin_abs_diff',
                      'bvps_abs_diff', 'ocfps_abs_diff', 'currentasset_turn_abs_diff', 'fix_assets_turn_abs_diff',
                      'nca_to_assets_abs_diff', 'op_to_tp_abs_diff', 'eqt_to_liab_abs_diff']  # Extended sample
        available_cols = [col for col in sample_cols if col in validation_df.columns]
        if available_cols:
            logger.debug(validation_df[available_cols].head().to_string(index=False))
    else:
        logger.warning("No overlapping periods for validation")

    # Save CSVs
    if not validation_df.empty:
        stock_name = stocks.replace(',', '_') if stocks else 'sample'
        validation_df.to_csv(f'{stock_name}_extended_validation.csv', index=False)
        computed_df.to_csv(f'{stock_name}_extended_computed.csv', index=False)
        logger.info(f"Saved: {stock_name}_extended_validation.csv & {stock_name}_extended_computed.csv")

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
        logger.info(f"Validation completed: {successful_stocks}/{total_stocks} stocks processed successfully ({success_rate:.1f}% success rate)")
    else:
        logger.info("Validation completed: No results to display")