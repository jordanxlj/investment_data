import os
import time
import datetime
from typing import Optional, List, Dict, Any
from functools import wraps

import fire
import pandas as pd
import numpy as np
import tushare as ts
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
import pymysql  # noqa: F401 - required by SQLAlchemy URL


# Tushare init
ts.set_token(os.environ["TUSHARE"])  # expects env var set
pro = ts.pro_api()


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator: Add retry mechanism to function

    Args:
        max_retries: Maximum number of retries
        delay: Initial delay time (seconds)
        backoff: Delay multiplier
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < max_retries:
                        print(f"Function {func.__name__} call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        print(f"Waiting {current_delay:.1f} seconds before retry...")
                        time.sleep(current_delay)
                        current_delay *= backoff  # exponential backoff
                    else:
                        print(f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise last_exception

            # This line won't be executed, but for type checker
            raise last_exception

        return wrapper
    return decorator


def call_tushare_api_with_retry(api_func, *args, **kwargs):
    """
    Generic Tushare API call function with retry mechanism

    Args:
        api_func: Tushare API function
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        DataFrame returned by API
    """
    @retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
    def _call_api():
        return api_func(*args, **kwargs)

    return _call_api()


TABLE_NAME = "ts_a_stock_financial_profile"


CREATE_TABLE_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
  ts_code                   VARCHAR(16)  NOT NULL,
  report_period             VARCHAR(10)  NOT NULL,
  period                    VARCHAR(8)   NOT NULL,
  currency                  VARCHAR(3)   NOT NULL,
  ann_date                  VARCHAR(10)  NULL,

  -- Income statement fields (based on actual Tushare field names)
  total_revenue             DECIMAL(16,4) NULL,
  operate_profit            DECIMAL(16,4) NULL,
  total_profit              DECIMAL(16,4) NULL,
  n_income                  DECIMAL(16,4) NULL,
  basic_eps                 FLOAT NULL,
  total_cogs                DECIMAL(16,4) NULL,
  sell_exp                  DECIMAL(16,4) NULL,
  admin_exp                 DECIMAL(16,4) NULL,
  fin_exp                   DECIMAL(16,4) NULL,
  invest_income             DECIMAL(16,4) NULL,
  interest_exp              DECIMAL(16,4) NULL,
  oper_exp                  DECIMAL(16,4) NULL,
  ebit                      DECIMAL(16,4) NULL,
  ebitda                    DECIMAL(16,4) NULL,
  income_tax                DECIMAL(16,4) NULL,

  -- Balance sheet fields
  total_assets              DECIMAL(16,4) NULL,
  total_liab                DECIMAL(16,4) NULL,
  total_hldr_eqy_inc_min_int DECIMAL(16,4) NULL,
  total_cur_assets          DECIMAL(16,4) NULL,
  total_cur_liab            DECIMAL(16,4) NULL,
  accounts_receiv           DECIMAL(16,4) NULL,
  inventories               DECIMAL(16,4) NULL,
  acct_payable              DECIMAL(16,4) NULL,
  fix_assets                DECIMAL(16,4) NULL,
  lt_borr                   DECIMAL(16,4) NULL,
  r_and_d                   DECIMAL(16,4) NULL,
  goodwill                  DECIMAL(16,4) NULL,
  intang_assets             DECIMAL(16,4) NULL,
  st_borr                   DECIMAL(16,4) NULL,
  total_share               DECIMAL(16,4) NULL,
  oth_eqt_tools_p_shr       DECIMAL(16,4) NULL,

  -- Cash flow statement fields
  n_cashflow_act            DECIMAL(16,4) NULL,
  n_cashflow_inv_act        DECIMAL(16,4) NULL,
  n_cash_flows_fnc_act      DECIMAL(16,4) NULL,
  free_cashflow             DECIMAL(16,4) NULL,
  c_pay_acq_const_fiolta    DECIMAL(16,4) NULL,
  c_fr_sale_sg              DECIMAL(16,4) NULL,
  c_paid_goods_s            DECIMAL(16,4) NULL,
  c_paid_to_for_empl        DECIMAL(16,4) NULL,
  c_paid_for_taxes          DECIMAL(16,4) NULL,
  n_incr_cash_cash_equ      DECIMAL(16,4) NULL,
  c_disp_withdrwl_invest    DECIMAL(16,4) NULL,
  c_pay_dist_dpcp_int_exp   DECIMAL(16,4) NULL,
  c_cash_equ_end_period     DECIMAL(16,4) NULL,

  -- === Financial indicator fields (grouped by relevance) ===

  -- 1. Basic financial indicators
  eps                       FLOAT NULL,
  dt_eps                    FLOAT NULL,
  gross_margin              FLOAT NULL,
  netprofit_margin          FLOAT NULL,
  grossprofit_margin        FLOAT NULL,
  ebitda_margin             FLOAT NULL,
  extra_item                DECIMAL(16,4) NULL,
  profit_dedt               DECIMAL(16,4) NULL,
  op_income                 DECIMAL(16,4) NULL,
  daa                       DECIMAL(16,4) NULL,
  rd_exp                    DECIMAL(16,4) NULL,

  -- 2. Solvency indicators
  current_ratio             FLOAT NULL,
  quick_ratio               FLOAT NULL,
  cash_ratio                FLOAT NULL,
  debt_to_assets            FLOAT NULL,
  assets_to_eqt             FLOAT NULL,
  dp_assets_to_eqt          FLOAT NULL,
  debt_to_eqt               FLOAT NULL,
  eqt_to_debt               FLOAT NULL,
  eqt_to_interestdebt       FLOAT NULL,
  ebit_to_interest          FLOAT NULL,
  ebitda_to_debt            FLOAT NULL,
  debt_to_assets_2          FLOAT NULL,
  assets_to_eqt_2           FLOAT NULL,
  dp_assets_to_eqt_2        FLOAT NULL,
  tangibleasset_to_debt     FLOAT NULL,
  tangasset_to_intdebt      FLOAT NULL,
  tangibleasset_to_netdebt  FLOAT NULL,

  -- 3. Operating efficiency indicators
  invturn_days              FLOAT NULL,
  arturn_days               FLOAT NULL,
  turn_days                 FLOAT NULL,
  inv_turn                  FLOAT NULL,
  ar_turn                   FLOAT NULL,
  ca_turn                   FLOAT NULL,
  fa_turn                   FLOAT NULL,
  assets_turn               FLOAT NULL,
  inventory_turnover        FLOAT NULL,
  inventory_days            FLOAT NULL,
  currentasset_turnover     FLOAT NULL,
  currentasset_days         FLOAT NULL,
  arturnover                FLOAT NULL,
  arturndays                FLOAT NULL,

  -- 4. Profitability indicators (ROE/ROA etc.)
  roic                      FLOAT NULL,
  roe_waa                   FLOAT NULL,
  roe_dt                    FLOAT NULL,
  roe_yearly                FLOAT NULL,
  roa                       FLOAT NULL,
  npta                      FLOAT NULL,
  npta_yearly               FLOAT NULL,
  roa_yearly                FLOAT NULL,
  roa_dp                    FLOAT NULL,
  roa_yearly_2              FLOAT NULL,
  roa_dp_2                  FLOAT NULL,
  roa_yearly_3              FLOAT NULL,
  roa_dp_3                  FLOAT NULL,

  -- 5. DuPont analysis indicators
  equity_multiplier         FLOAT NULL,
  roe_waa_2                 FLOAT NULL,
  roe_avg                   FLOAT NULL,
  roe_waa_2_dedt            FLOAT NULL,
  roe_avg_dedt              FLOAT NULL,
  roe_waa_2_nonr            FLOAT NULL,
  roe_avg_nonr              FLOAT NULL,
  roe_waa_2_dedt_ttm        FLOAT NULL,
  roe_dt_2                  FLOAT NULL,
  debt_to_equity_1          FLOAT NULL,
  equity_ratio              FLOAT NULL,

  -- 6. Per share indicators
  total_revenue_ps          FLOAT NULL,
  revenue_ps                FLOAT NULL,
  capital_rese_ps           FLOAT NULL,
  surplus_rese_ps           FLOAT NULL,
  undist_profit_ps          FLOAT NULL,
  bps                       FLOAT NULL,
  ocfps                     FLOAT NULL,
  retainedps                FLOAT NULL,
  cfps                      FLOAT NULL,
  ebit_ps                   FLOAT NULL,
  fcff_ps                   FLOAT NULL,
  fcfe_ps                   FLOAT NULL,
  q_eps                     FLOAT NULL,

  -- 7. Cash flow indicators
  cf_sales                  FLOAT NULL,
  cf_nm                     FLOAT NULL,
  cf_liabs                  FLOAT NULL,
  cashflow_m                FLOAT NULL,
  op_of_gr                  FLOAT NULL,
  ocf_to_debt               FLOAT NULL,
  ocf_to_interestdebt       FLOAT NULL,
  ocf_to_netdebt            FLOAT NULL,
  ocf_to_shortdebt          FLOAT NULL,
  ocf_to_or                 FLOAT NULL,
  ocf_to_opincome           FLOAT NULL,
  salescash_to_or           FLOAT NULL,
  q_ocf_to_sales            FLOAT NULL,

  -- 8. Growth indicators (YoY growth rates)
  basic_eps_yoy             FLOAT NULL,
  dt_eps_yoy                FLOAT NULL,
  cfps_yoy                  FLOAT NULL,
  op_yoy                    FLOAT NULL,
  ebt_yoy                   FLOAT NULL,
  netprofit_yoy             FLOAT NULL,
  dt_netprofit_yoy          FLOAT NULL,
  ocf_yoy                   FLOAT NULL,
  roe_yoy                   FLOAT NULL,
  bps_yoy                   FLOAT NULL,
  assets_yoy                FLOAT NULL,
  eqt_yoy                   FLOAT NULL,
  tr_yoy                    FLOAT NULL,
  or_yoy                    FLOAT NULL,
  equity_yoy                FLOAT NULL,

  -- 9. Quarterly financial indicators
  q_opincome                DECIMAL(16,4) NULL,
  q_investincome            DECIMAL(16,4) NULL,
  q_dtprofit                DECIMAL(16,4) NULL,
  q_netprofit_margin        FLOAT NULL,
  q_gsprofit_margin         FLOAT NULL,
  q_exp_to_sales            FLOAT NULL,
  q_profit_to_gr            FLOAT NULL,
  q_saleexp_to_gr           FLOAT NULL,
  q_adminexp_to_gr          FLOAT NULL,
  q_finaexp_to_gr           FLOAT NULL,
  q_impair_to_gr_ttm        FLOAT NULL,
  q_gc_to_gr                FLOAT NULL,
  q_op_to_gr                FLOAT NULL,
  q_roe                     FLOAT NULL,
  q_dt_roe                  FLOAT NULL,
  q_npta                    FLOAT NULL,

  -- 10. Quarterly sequential growth rates
  q_gr_yoy                  FLOAT NULL,
  q_gr_qoq                  FLOAT NULL,
  q_sales_yoy               FLOAT NULL,
  q_sales_qoq               FLOAT NULL,
  q_op_yoy                  FLOAT NULL,
  q_op_qoq                  FLOAT NULL,
  q_profit_yoy              FLOAT NULL,
  q_profit_qoq              FLOAT NULL,
  q_netprofit_yoy           FLOAT NULL,
  q_netprofit_qoq           FLOAT NULL,

  -- 11. Cost and expense structure analysis
  cogs_of_sales             FLOAT NULL,
  expense_of_sales          FLOAT NULL,
  profit_to_gr              FLOAT NULL,
  saleexp_to_gr             FLOAT NULL,
  adminexp_to_gr            FLOAT NULL,
  finaexp_to_gr             FLOAT NULL,
  impair_to_gr_ttm          FLOAT NULL,
  gc_of_gr                  FLOAT NULL,
  op_to_ebt                 FLOAT NULL,
  tax_to_ebt                FLOAT NULL,
  dtprofit_to_profit        FLOAT NULL,
  profit_to_op              FLOAT NULL,
  profit_prefin_exp         DECIMAL(16,4) NULL,
  non_op_profit             DECIMAL(16,4) NULL,

  -- 12. Asset structure analysis
  ca_to_assets              FLOAT NULL,
  nca_to_assets             FLOAT NULL,
  tbassets_to_totalassets   FLOAT NULL,
  fixed_assets              DECIMAL(16,4) NULL,
  int_to_talcap             FLOAT NULL,
  eqt_to_talcapital         FLOAT NULL,
  currentdebt_to_debt       FLOAT NULL,
  longdeb_to_debt           FLOAT NULL,
  longdebt_to_workingcapital FLOAT NULL,
  capitalized_to_da         FLOAT NULL,

  -- 13. Valuation indicators
  current_exint             FLOAT NULL,
  non_current_exint         FLOAT NULL,
  intrinsicvalue            DECIMAL(16,4) NULL,
  tmv                       DECIMAL(16,4) NULL,
  lmv                       DECIMAL(16,4) NULL,

  PRIMARY KEY (ts_code, report_period),
  INDEX idx_report_period (report_period),
  INDEX idx_period (period),
  INDEX idx_ts_code (ts_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;
"""


# === Data source field grouping definitions ===

# Base fields (shared by all data sources)
BASE_COLUMNS = ["ts_code", "ann_date", "report_period", "period", "currency"]

# Income statement fields
INCOME_COLUMNS = [
    "total_revenue", "operate_profit", "total_profit", "n_income", "basic_eps",
    "total_cogs", "sell_exp", "admin_exp", "fin_exp", "invest_income", "interest_exp",
    "oper_exp", "ebit", "ebitda", "income_tax"
]

# Balance sheet fields
BALANCE_COLUMNS = [
    "total_assets", "total_liab", "total_hldr_eqy_inc_min_int", "total_cur_assets",
    "total_cur_liab", "accounts_receiv", "inventories", "acct_payable",
    "fix_assets", "lt_borr", "r_and_d", "goodwill", "intang_assets", "st_borr",
    "total_share", "oth_eqt_tools_p_shr"
]

# Cash flow statement fields
CASHFLOW_COLUMNS = [
    "n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act",
    "free_cashflow", "c_pay_acq_const_fiolta", "c_fr_sale_sg", "c_paid_goods_s",
    "c_paid_to_for_empl", "c_paid_for_taxes", "n_incr_cash_cash_equ",
    "c_disp_withdrwl_invest", "c_pay_dist_dpcp_int_exp", "c_cash_equ_end_period"
]

# Financial indicator fields (grouped by relevance)
INDICATOR_COLUMNS = [
    # Basic financial indicators
    "eps", "dt_eps", "gross_margin", "netprofit_margin", "grossprofit_margin",
    "ebitda_margin", "extra_item", "profit_dedt", "op_income", "daa", "rd_exp",

    # Solvency indicators
    "current_ratio", "quick_ratio", "cash_ratio", "debt_to_assets", "assets_to_eqt",
    "dp_assets_to_eqt", "debt_to_eqt", "eqt_to_debt", "eqt_to_interestdebt",
    "ebit_to_interest", "ebitda_to_debt", "debt_to_assets_2", "assets_to_eqt_2",
    "dp_assets_to_eqt_2", "tangibleasset_to_debt", "tangasset_to_intdebt",
    "tangibleasset_to_netdebt",

    # Operating efficiency indicators
    "invturn_days", "arturn_days", "turn_days", "inv_turn", "ar_turn", "ca_turn",
    "fa_turn", "assets_turn", "inventory_turnover", "inventory_days",
    "currentasset_turnover", "currentasset_days", "arturnover", "arturndays",

    # Profitability indicators (ROE/ROA etc.)
    "roic", "roe_waa", "roe_dt", "roe_yearly", "roa", "npta", "npta_yearly",
    "roa_yearly", "roa_dp", "roa_yearly_2", "roa_dp_2", "roa_yearly_3", "roa_dp_3",

    # DuPont analysis indicators
    "equity_multiplier", "roe_waa_2", "roe_avg", "roe_waa_2_dedt", "roe_avg_dedt",
    "roe_waa_2_nonr", "roe_avg_nonr", "roe_waa_2_dedt_ttm", "roe_dt_2",
    "debt_to_equity_1", "equity_ratio",

    # Per share indicators
    "total_revenue_ps", "revenue_ps", "capital_rese_ps", "surplus_rese_ps",
    "undist_profit_ps", "bps", "ocfps", "retainedps", "cfps", "ebit_ps",
    "fcff_ps", "fcfe_ps", "q_eps",

    # Cash flow indicators
    "cf_sales", "cf_nm", "cf_liabs", "cashflow_m", "op_of_gr", "ocf_to_debt",
    "ocf_to_interestdebt", "ocf_to_netdebt", "ocf_to_shortdebt", "ocf_to_or",
    "ocf_to_opincome", "salescash_to_or", "q_ocf_to_sales",

    # Growth indicators (YoY growth rates)
    "basic_eps_yoy", "dt_eps_yoy", "cfps_yoy", "op_yoy", "ebt_yoy", "netprofit_yoy",
    "dt_netprofit_yoy", "ocf_yoy", "roe_yoy", "bps_yoy", "assets_yoy", "eqt_yoy",
    "tr_yoy", "or_yoy", "equity_yoy",

    # Quarterly financial indicators
    "q_opincome", "q_investincome", "q_dtprofit", "q_netprofit_margin",
    "q_gsprofit_margin", "q_exp_to_sales", "q_profit_to_gr", "q_saleexp_to_gr",
    "q_adminexp_to_gr", "q_finaexp_to_gr", "q_impair_to_gr_ttm", "q_gc_to_gr",
    "q_op_to_gr", "q_roe", "q_dt_roe", "q_npta",

    # Quarterly sequential growth rates
    "q_gr_yoy", "q_gr_qoq", "q_sales_yoy", "q_sales_qoq", "q_op_yoy", "q_op_qoq",
    "q_profit_yoy", "q_profit_qoq", "q_netprofit_yoy", "q_netprofit_qoq",

    # Cost and expense structure analysis
    "cogs_of_sales", "expense_of_sales", "profit_to_gr", "saleexp_to_gr",
    "adminexp_to_gr", "finaexp_to_gr", "impair_to_gr_ttm", "gc_of_gr",
    "op_to_ebt", "tax_to_ebt", "dtprofit_to_profit", "profit_to_op",
    "profit_prefin_exp", "non_op_profit",

    # Asset structure analysis
    "ca_to_assets", "nca_to_assets", "tbassets_to_totalassets", "fixed_assets",
    "int_to_talcap", "eqt_to_talcapital", "currentdebt_to_debt", "longdeb_to_debt",
    "longdebt_to_workingcapital", "capitalized_to_da",

    # Valuation indicators
    "current_exint", "non_current_exint", "intrinsicvalue", "tmv", "lmv"
]

# === Data source field configuration ===

# API field name list (all three major financial statements contain these base fields)
# Note: API returns 'end_date' but database stores as 'ann_date'
API_COMMON_FIELDS = ['ts_code', 'ann_date', 'end_date', 'report_type']

# Financial indicators base fields (does not include report_type)
INDICATOR_BASE_FIELDS = ['ts_code', 'ann_date', 'end_date']  # Keep end_date for API call, will be mapped later

# === Merged total field list (used for database operations) ===
ALL_COLUMNS: List[str] = BASE_COLUMNS + INCOME_COLUMNS + BALANCE_COLUMNS + CASHFLOW_COLUMNS + INDICATOR_COLUMNS


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all expected columns exist and normalize data types"""
    # Ensure all expected columns exist
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Keep only known columns, in order
    out = df[ALL_COLUMNS].copy()

    # Normalize types
    if not out.empty:
        # String columns
        string_cols = ["ts_code", "report_period", "period", "currency", "ann_date"]
        for col in string_cols:
            if col in out.columns:
                out[col] = out[col].astype(str).replace('nan', None).replace('None', None)

        # Numeric columns - convert to float first, then handle None values
        numeric_cols = [col for col in ALL_COLUMNS if col not in string_cols]
        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        # Ensure DB NULLs: cast to object then replace NaN with None
        out = out.astype(object).where(pd.notna(out), None)
        # Extra safety for numpy.nan
        out = out.replace({np.nan: None})

    return out


def _fetch_single_period_data(report_period: str) -> pd.DataFrame:
    """
    Fetch financial data for a single period

    ✨ Optimization features:
    - Retry mechanism: Each API call retries up to 3 times with exponential backoff
    - Smart merging: Merge all data sources at once, supports partial data
    - Error recovery: Failure of one data source doesn't affect others
    - Detailed logging: Complete data fetching and merging process records
    - Simplified configuration: Field names identical to API, no mapping table needed

    Data merging strategy:
    1. Three major financial statements (Income + Balance + Cash Flow) use common keys for merging
    2. Financial indicator data uses simplified keys for merging (no report_type field)
    3. Supports partial data scenarios, returns results even with only partial data sources

    Field configuration optimization:
    - COMMON_FIELDS: Common fields for three major statements ['ts_code', 'ann_date', 'end_date', 'report_type']
    - INDICATOR_BASE_FIELDS: Base fields for financial indicators ['ts_code', 'ann_date', 'end_date']
    - Use field lists directly, no redundant mapping table

    Args:
        report_period: Report period, format like '20231231'

    Returns:
        Merged financial data DataFrame containing all available data
    """
    try:
        print(f"Fetching financial data for period: {report_period}")

        # 1. Get income statement data (with retry mechanism)
        income_fields = API_COMMON_FIELDS + INCOME_COLUMNS
        income_df = call_tushare_api_with_retry(
            pro.income_vip,
            ts_code='000001.SZ',
            period=report_period,
            fields=','.join(income_fields)
        )

        # 2. Get balance sheet data (with retry mechanism)
        balance_fields = API_COMMON_FIELDS + BALANCE_COLUMNS
        balance_df = call_tushare_api_with_retry(
            pro.balancesheet_vip,
            ts_code='000001.SZ',
            period=report_period,
            fields=','.join(balance_fields)
        )

        # 3. Get cash flow statement data (with retry mechanism)
        cashflow_fields = API_COMMON_FIELDS + CASHFLOW_COLUMNS
        cashflow_df = call_tushare_api_with_retry(
            pro.cashflow_vip,
            ts_code='000001.SZ',
            period=report_period,
            fields=','.join(cashflow_fields)
        )

        # 4. Get financial indicators data (with retry mechanism)
        indicator_fields = INDICATOR_BASE_FIELDS + INDICATOR_COLUMNS
        indicator_df = call_tushare_api_with_retry(
            pro.fina_indicator_vip,
            ts_code='000001.SZ',
            period=report_period,
            fields=','.join(indicator_fields)
        )

        # === Merge data by source grouping ===

        # Check availability of each data source
        data_sources = {
            'income_statement': income_df,
            'balance_sheet': balance_df,
            'cash_flow': cashflow_df,
            'financial_indicators': indicator_df
        }

        available_sources = {name: df for name, df in data_sources.items() if not df.empty}
        if not available_sources:
            print(f"No data available for period {report_period}")
            return pd.DataFrame()

        print(f"Available data sources for period {report_period}: {', '.join(available_sources.keys())}")

        # Debug: Print record counts for each data source
        for name, df in available_sources.items():
            print(f"  {name}: {len(df)} records")
            if len(df) > 0:
                print(f"    Sample ts_code: {df['ts_code'].iloc[0] if 'ts_code' in df.columns else 'N/A'}")
                print(f"    Sample ann_date: {df['ann_date'].iloc[0] if 'ann_date' in df.columns else 'N/A'}")
                print(f"    Sample end_date: {df['end_date'].iloc[0] if 'end_date' in df.columns else 'N/A'}")

                # Check for duplicates within each data source
                if len(df) > 1:
                    duplicate_keys = df.groupby(['ts_code', 'ann_date', 'end_date']).size()
                    duplicates = duplicate_keys[duplicate_keys > 1]
                    if len(duplicates) > 0:
                        print(f"    ⚠️  Found duplicates in {name}:")
                        for (ts_code, ann_date, end_date), count in duplicates.items():
                            print(f"      {ts_code} {ann_date} {end_date}: {count} duplicates")

                    # Show all records if there are duplicates
                    if len(df) <= 3:  # Only show if not too many records
                        print(f"    All records in {name}:")
                        for i, row in df.iterrows():
                            report_type = row.get('report_type', 'N/A')
                            print(f"      Record {i}: ts_code={row['ts_code']}, ann_date={row['ann_date']}, end_date={row['end_date']}, report_type={report_type}")

                            # Show first 5 financial data columns to check for differences
                            financial_cols = [col for col in df.columns if col not in ['ts_code', 'ann_date', 'end_date', 'report_type']]
                            if financial_cols:
                                sample_data = []
                                for col in financial_cols[:5]:  # Show first 5 financial columns
                                    if col in row.index and pd.notna(row[col]):
                                        sample_data.append(f"{col}={row[col]}")
                                if sample_data:
                                    print(f"        Financial data: {', '.join(sample_data[:3])}...")
                                    if len(sample_data) > 3:
                                        print(f"        ... and {len(sample_data) - 3} more fields")

                        # Check if records are exactly identical
                        if len(df) == 2:
                            record1 = df.iloc[0]
                            record2 = df.iloc[1]
                            are_identical = record1.equals(record2)
                            print(f"    Records are exactly identical: {are_identical}")

                            if not are_identical:
                                # Find differing columns
                                differing_cols = []
                                for col in df.columns:
                                    if not pd.isna(record1[col]) or not pd.isna(record2[col]):
                                        if str(record1[col]) != str(record2[col]):
                                            differing_cols.append(col)

                                if differing_cols:
                                    print(f"    Differing columns: {differing_cols}")
                                    for col in differing_cols[:5]:  # Show first 5 differences
                                        print(f"      {col}: '{record1[col]}' vs '{record2[col]}'")

                # Remove duplicates within each data source before merging
                initial_len = len(df)
                df = df.drop_duplicates(subset=['ts_code', 'ann_date', 'end_date'], keep='first')
                if len(df) < initial_len:
                    print(f"    🧹 Removed {initial_len - len(df)} duplicates from {name}")

                available_sources[name] = df

        # Merge strategy: connect by data source grouping
        merged_df = None

        # 1. Merge three major financial statements (Income + Balance + Cash Flow)
        # These data sources have the same join keys: ['ts_code', 'ann_date', 'end_date', 'report_type']
        financial_statements = []
        if 'income_statement' in available_sources:
            financial_statements.append(available_sources['income_statement'])
        if 'balance_sheet' in available_sources:
            financial_statements.append(available_sources['balance_sheet'])
        if 'cash_flow' in available_sources:
            financial_statements.append(available_sources['cash_flow'])

        if financial_statements:
            try:
                # Start from the first data source and gradually merge other data sources
                merged_df = financial_statements[0]
                print(f"Initial merge base: {len(merged_df)} records from {list(available_sources.keys())[0]}")

                for i, df in enumerate(financial_statements[1:], 1):
                    before_count = len(merged_df)
                    source_name = list(available_sources.keys())[i]

                    # Check merge keys before merging
                    merge_keys_check = merged_df.groupby(API_COMMON_FIELDS).size()
                    incoming_keys_check = df.groupby(API_COMMON_FIELDS).size()

                    print(f"  Merge step {i} ({source_name}):")
                    print(f"    Before: {before_count} records")
                    print(f"    Adding: {len(df)} records")
                    print(f"    Merge keys in base: {len(merge_keys_check)} unique combinations")
                    print(f"    Merge keys incoming: {len(incoming_keys_check)} unique combinations")

                    merged_df = merged_df.merge(
                        df,
                        on=API_COMMON_FIELDS,
                        how='outer'
                    )
                    after_count = len(merged_df)
                    print(f"    After: {after_count} records (+{after_count - before_count})")

                print(f"Successfully merged {len(financial_statements)} financial statements: {len(merged_df)} records")
            except Exception as e:
                print(f"Error merging financial statements: {e}")
                return pd.DataFrame()

        # 2. Merge financial indicators data
        # Financial indicators data doesn't have 'report_type' field, use simplified join keys
        if merged_df is not None and 'financial_indicators' in available_sources:
            try:
                before_count = len(merged_df)
                indicator_df = available_sources['financial_indicators']

                # Check merge keys
                base_keys = merged_df.groupby(['ts_code', 'ann_date', 'end_date']).size()
                indicator_keys = indicator_df.groupby(['ts_code', 'ann_date', 'end_date']).size()

                print(f"Financial indicators merge:")
                print(f"  Base data: {len(base_keys)} unique combinations, {before_count} total records")
                print(f"  Indicators: {len(indicator_keys)} unique combinations, {len(indicator_df)} total records")

                merged_df = merged_df.merge(
                    indicator_df,
                    on=['ts_code', 'ann_date', 'end_date'],
                    how='left'  # Left join to ensure financial data completeness
                )
                after_count = len(merged_df)
                print(f"  Result: {after_count} records ({'+' if after_count > before_count else ''}{after_count - before_count})")
                print(f"Successfully merged financial indicators: {len(merged_df)} records total")
            except Exception as e:
                print(f"Error merging financial indicators: {e}")
                print("Continuing with financial statements only...")

        # Handle edge case: only financial indicators data, no major financial statements
        elif merged_df is None and 'financial_indicators' in available_sources:
            merged_df = available_sources['financial_indicators'].copy()
            print(f"Using financial indicators only: {len(merged_df)} records")

        if merged_df is None:
            print(f"No usable data after merging for period {report_period}")
            return pd.DataFrame()

        print(f"merged_df: {merged_df}")
        # Add unified fields
        try:
            merged_df['ts_code'] = merged_df['ts_code']
            merged_df['ann_date'] = merged_df['end_date']  # Map API end_date to database ann_date
            merged_df['report_period'] = merged_df['end_date'].astype(str).str[:4] + '-' + merged_df['end_date'].astype(str).str[4:6] + '-' + merged_df['end_date'].astype(str).str[6:8]
            merged_df['period'] = 'annual' if report_period.endswith('1231') else 'quarter'
            merged_df['currency'] = 'CNY'  # A-share default currency is CNY

            # Remove API-specific fields that don't exist in database
            if 'end_date' in merged_df.columns:
                merged_df = merged_df.drop('end_date', axis=1)

            # Remove duplicates based on primary key (ts_code, report_period)
            initial_count = len(merged_df)

            # Debug: Check for duplicates before removal
            duplicate_check = merged_df.groupby(['ts_code', 'report_period']).size()
            duplicates_found = duplicate_check[duplicate_check > 1]
            if len(duplicates_found) > 0:
                print(f"Found {len(duplicates_found)} duplicate groups before removal:")
                for (ts_code, report_period), count in duplicates_found.items():
                    print(f"  {ts_code} {report_period}: {count} duplicates")

            merged_df = merged_df.drop_duplicates(subset=['ts_code', 'report_period'], keep='first')
            final_count = len(merged_df)

            if initial_count != final_count:
                print(f"Removed {initial_count - final_count} duplicate records, kept {final_count} unique records")
            else:
                print(f"No duplicates found, kept {final_count} records")

            print(f"Successfully processed {len(merged_df)} financial records for period {report_period}")
            return merged_df

        except Exception as e:
            print(f"Error processing unified fields: {e}")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error in _fetch_single_period_data for period {report_period}: {e}")
        return pd.DataFrame()


def _generate_periods(end_date: str, period: str = "annual", limit: int = 1) -> List[str]:
    """Generate list of periods that need to be processed"""
    periods = []
    end_dt = datetime.datetime.strptime(end_date, "%Y%m%d")

    if period == "annual":
        # Annual reports: Get the most recent years' 12-31
        for i in range(limit):
            year = end_dt.year - i
            if datetime.datetime(year, 12, 31) <= end_dt:
                periods.append(f"{year}1231")
    else:
        # Quarterly reports: Get the most recent quarters' end dates
        quarters = [(3, 31), (6, 30), (9, 30), (12, 31)]
        current_quarter = None

        # Find current quarter
        for month, day in reversed(quarters):
            q_date = datetime.datetime(end_dt.year, month, day)
            if q_date <= end_dt:
                current_quarter = q_date
                break

        if current_quarter:
            for i in range(limit):
                periods.append(current_quarter.strftime("%Y%m%d"))
                # Calculate previous quarter
                if current_quarter.month == 3:
                    current_quarter = datetime.datetime(current_quarter.year - 1, 12, 31)
                elif current_quarter.month == 6:
                    current_quarter = datetime.datetime(current_quarter.year, 3, 31)
                elif current_quarter.month == 9:
                    current_quarter = datetime.datetime(current_quarter.year, 6, 30)
                else:  # 12
                    current_quarter = datetime.datetime(current_quarter.year, 9, 30)

    return periods


def _fetch_financial_data(end_date: str, period: str = "annual", limit: int = 1) -> pd.DataFrame:
    """Fetch financial data for multiple periods (maintain compatibility)"""
    periods = _generate_periods(end_date, period, limit)

    if not periods:
        print("No valid periods found")
        return pd.DataFrame()

    print(f"Fetching financial data for periods: {periods}")

    all_data = []
    for report_period in periods:
        df = _fetch_single_period_data(report_period)
        if not df.empty:
            all_data.append(df)

        # Add delay to avoid API limits
        time.sleep(0.5)

    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        print(f"Successfully fetched {len(result_df)} financial records in total")
        return result_df
    else:
        return pd.DataFrame()


def _upsert_batch(engine, df: pd.DataFrame, chunksize: int = 1000) -> int:
    """Batch upsert data to MySQL

    Returns:
        int: Number of records processed (not necessarily inserted/updated)
    """
    if df is None or df.empty:
        return 0

    # Return the actual number of records in the DataFrame, not the SQL result rowcount
    # This gives a more accurate representation of processed records
    total_processed = len(df)

    from sqlalchemy import Table, MetaData
    meta = MetaData()
    table = Table(TABLE_NAME, meta, autoload_with=engine)

    rows = df.to_dict(orient="records")
    total_affected = 0

    with engine.begin() as conn:
        for i in range(0, len(rows), chunksize):
            batch = rows[i:i+chunksize]
            stmt = mysql_insert(table).values(batch)
            update_map: Dict[str, Any] = {
                c: getattr(stmt.inserted, c)
                for c in ALL_COLUMNS
                if c not in ("ts_code", "report_period")
            }
            ondup = stmt.on_duplicate_key_update(**update_map)
            result = conn.execute(ondup)
            total_affected += result.rowcount or 0

    # Log both metrics for transparency
    print(f"Processed {total_processed} records, database reported {total_affected} affected rows")
    return total_processed


def update_a_stock_financial_profile(
    mysql_url: str = "mysql+pymysql://root:@127.0.0.1:3306/investment_data",
    end_date: Optional[str] = None,
    period: str = "quarter",
    limit: int = 10,
    chunksize: int = 1000,
) -> None:
    """
    Incrementally fetch Tushare financial profile data and write to MySQL ts_a_stock_financial_profile table

    ✨ Optimization features:
    - Period-by-period processing: Avoid memory overflow from loading too much data at once
    - Real-time writing: Write to database immediately after processing each period
    - Memory-friendly: Release memory immediately after processing each period's data
    - 🔄 Retry mechanism: Automatically retry API calls up to 3 times with exponential backoff

    Contains complete financial data, grouped by relevance:
    1. Three major financial statements: Income statement, Balance sheet, Cash flow statement
    2. Basic financial indicators: Gross margin, net margin and other basic indicators
    3. Solvency indicators: Current ratio, quick ratio, debt-to-assets ratio, etc.
    4. Operating efficiency indicators: Turnover ratios, operating efficiency, etc.
    5. Profitability indicators: ROE, ROA, net margin, etc.
    6. DuPont analysis indicators: Equity multiplier, ROE decomposition, etc.
    7. Per share indicators: Earnings per share, book value per share, etc.
    8. Cash flow indicators: Cash flow ratios, cash flow efficiency, etc.
    9. Growth indicators: Various business growth rates
    10. Quarterly financial indicators: Quarterly data and ratios
    11. Cost and expense structure analysis: Expense ratio analysis
    12. Asset structure analysis: Asset allocation and capital structure
    13. Valuation indicators: Market value, intrinsic value, etc.

    Data sources:
    - pro.income_vip() - Income statement
    - pro.balancesheet_vip() - Balance sheet
    - pro.cashflow_vip() - Cash flow statement
    - pro.fina_indicator_vip() - Financial indicators

    Data type optimization:
    - Ratios and per-share earnings: FLOAT type (precise to 6 decimal places)
    - Absolute amounts: DECIMAL(16,4) type (precise to cent)

    Field configuration optimization:
    - Field names identical to Tushare API, no mapping table needed
    - Use COMMON_FIELDS and INDICATOR_BASE_FIELDS for simplified configuration
    - Four major data source groups: Income statement, Balance sheet, Cash flow, Financial indicators

    Processing workflow:
    1. Generate list of periods that need processing
    2. Process each period individually:
       - Fetch four major data sources (three major statements + financial indicators)
       - Smart merging: Merge all available data at once
       - Error recovery: Partial data failure doesn't affect the whole process
       - Write to database immediately to free up memory
    3. Statistical processing results

    Data merging optimization:
    - One-time merging strategy: Three major statements → Financial indicators
    - Smart key matching: Select appropriate join keys based on data source characteristics
    - Partial data support: Continue even if some data sources fail
    - Detailed status monitoring: Real-time display of data fetching and merging status

    Retry mechanism details:
    - Each API call failure automatically retries, up to 3 times
    - Retry intervals: 1s → 2s → 4s (exponential backoff)
    - Detailed recording of error information for each retry
    - Failure of one API doesn't affect other API calls

    Args:
        mysql_url: MySQL connection URL
        end_date: End date in YYYYMMDD format, defaults to yesterday
        period: Report period type, 'annual' or 'quarter'
        limit: Limit on number of report periods to fetch
        chunksize: Batch processing size
    """

    # Set end date
    if end_date is None:
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        end_date = yesterday.strftime("%Y%m%d")

    print(f"Starting to update financial profile data, end date: {end_date}, period type: {period}, limit: {limit}")

    # Create database engine
    engine = create_engine(mysql_url, pool_recycle=3600)

    # Create table structure
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_DDL))

    # Fetch and process financial profile data period by period
    print("Fetching and processing financial profile data period by period...")

    periods = _generate_periods(end_date, period, limit)
    if not periods:
        print("No valid periods found")
        return

    total_written = 0
    for i, report_period in enumerate(periods):
        print(f"\nProcessing period {i+1}/{len(periods)}: {report_period}")

        # Get data for single period
        df = _fetch_single_period_data(report_period)

        if df.empty:
            print(f"No data retrieved for period {report_period}, skipping")
            continue

        # Data normalization
        df = _coerce_schema(df)

        print(f"Retrieved {len(df)} financial profile records for period {report_period}")

        # Immediately upsert to database
        written = _upsert_batch(engine, df, chunksize=chunksize)
        total_written += written

        print(f"Period {report_period} processing completed, {written} records written")

        # Add delay to avoid API limits (already added in _fetch_single_period_data)
        if i < len(periods) - 1:  # Not the last period, add delay
            time.sleep(0.5)

    print(f"\nUpdate completed, processed {len(periods)} periods, total {total_written} records written")


if __name__ == "__main__":
    # Run main program directly, test and tool functions please use independent modules
    fire.Fire(update_a_stock_financial_profile)
