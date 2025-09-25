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
from sqlalchemy import Table, MetaData
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
  report_period             DATE         NOT NULL,
  period                    VARCHAR(8)   NOT NULL,
  currency                  VARCHAR(3)   NOT NULL,
  ann_date                  DATE         NULL,

  -- Income statement fields (万元存储 - converted from 元)
  total_revenue             DECIMAL(16,4) NULL COMMENT '总营收(万元)',
  revenue                   DECIMAL(16,4) NULL COMMENT '营业收入(万元)',
  operate_profit            DECIMAL(16,4) NULL COMMENT '营业利润(万元)',
  total_profit              DECIMAL(16,4) NULL COMMENT '利润总额(万元)',
  n_income_attr_p           DECIMAL(16,4) NULL COMMENT '净利润(万元)',
  basic_eps                 FLOAT NULL COMMENT '基本每股收益(元)',
  total_cogs                DECIMAL(16,4) NULL COMMENT '营业总成本(万元)',
  oper_cost                 DECIMAL(16,4) NULL COMMENT '营业成本(万元)',
  sell_exp                  DECIMAL(16,4) NULL COMMENT '销售费用(万元)',
  admin_exp                 DECIMAL(16,4) NULL COMMENT '管理费用(万元)',
  fin_exp                   DECIMAL(16,4) NULL COMMENT '财务费用(万元)',
  invest_income             DECIMAL(16,4) NULL COMMENT '投资收益(万元)',
  interest_exp              DECIMAL(16,4) NULL COMMENT '利息支出(万元)',
  oper_exp                  DECIMAL(16,4) NULL COMMENT '营业支出(万元)',
  ebit                      DECIMAL(16,4) NULL COMMENT '息税前利润(万元)',
  ebitda                    DECIMAL(16,4) NULL COMMENT 'EBITDA(万元)',
  income_tax                DECIMAL(16,4) NULL COMMENT '所得税(万元)',
  comshare_payable_dvd      DECIMAL(16,4) NULL COMMENT '应付股利(万元)',

  -- Balance sheet fields (万元存储 - converted from 元)
  total_assets              DECIMAL(16,4) NULL COMMENT '总资产(万元)',
  total_liab                DECIMAL(16,4) NULL COMMENT '总负债(万元)',
  total_equity              DECIMAL(16,4) NULL COMMENT '股东权益(万元)',
  total_hldr_eqy_exc_min_int DECIMAL(16,4) NULL COMMENT '股东权益(不含少数股东权益)(万元)',
  total_hldr_eqy_inc_min_int DECIMAL(16,4) NULL COMMENT '股东权益(含少数股东权益)(万元)',
  total_cur_assets          DECIMAL(16,4) NULL COMMENT '流动资产(万元)',
  total_cur_liab            DECIMAL(16,4) NULL COMMENT '流动负债(万元)',
  accounts_receiv           DECIMAL(16,4) NULL COMMENT '应收账款(万元)',
  inventories               DECIMAL(16,4) NULL COMMENT '存货(万元)',
  acct_payable              DECIMAL(16,4) NULL COMMENT '应付账款(万元)',
  fix_assets                DECIMAL(16,4) NULL COMMENT '固定资产(万元)',
  lt_borr                   DECIMAL(16,4) NULL COMMENT '长期借款(万元)',
  r_and_d                   DECIMAL(16,4) NULL COMMENT '研发支出(万元)',
  goodwill                  DECIMAL(16,4) NULL COMMENT '商誉(万元)',
  intang_assets             DECIMAL(16,4) NULL COMMENT '无形资产(万元)',
  st_borr                   DECIMAL(16,4) NULL COMMENT '短期借款(万元)',
  total_share               DECIMAL(16,4) NULL COMMENT '股本(万元)',
  oth_eqt_tools_p_shr       DECIMAL(16,4) NULL COMMENT '其他权益工具(万元)',

  -- Cash flow statement fields (万元存储 - converted from 元)
  n_cashflow_act            DECIMAL(16,4) NULL COMMENT '经营现金流(万元)',
  n_cashflow_inv_act        DECIMAL(16,4) NULL COMMENT '投资现金流(万元)',
  n_cash_flows_fnc_act      DECIMAL(16,4) NULL COMMENT '融资现金流(万元)',
  free_cashflow             DECIMAL(16,4) NULL COMMENT '自由现金流(万元)',
  c_pay_acq_const_fiolta    DECIMAL(16,4) NULL COMMENT '购建固定资产(万元)',
  c_fr_sale_sg              DECIMAL(16,4) NULL COMMENT '销售商品收款(万元)',
  c_paid_goods_s            DECIMAL(16,4) NULL COMMENT '购买商品付款(万元)',
  c_paid_to_for_empl        DECIMAL(16,4) NULL COMMENT '支付职工薪酬(万元)',
  c_paid_for_taxes          DECIMAL(16,4) NULL COMMENT '支付税费(万元)',
  n_incr_cash_cash_equ      DECIMAL(16,4) NULL COMMENT '现金等价物净增加(万元)',
  c_disp_withdrwl_invest    DECIMAL(16,4) NULL COMMENT '处置投资收款(万元)',
  c_pay_dist_dpcp_int_exp   DECIMAL(16,4) NULL COMMENT '分配股利利息(万元)',
  c_cash_equ_end_period     DECIMAL(16,4) NULL COMMENT '期末现金余额(万元)',

  -- === Financial indicator fields (grouped by relevance) ===

  -- 1. Basic financial indicators
  eps                       FLOAT NULL COMMENT '每股收益(元)',
  dt_eps                    FLOAT NULL COMMENT '稀释每股收益(元)',
  netprofit_margin          FLOAT NULL COMMENT '净利率(%)',
  grossprofit_margin        FLOAT NULL COMMENT '毛利润率(%)',
  ebitda_margin             FLOAT NULL COMMENT 'EBITDA利润率(%)',
  gross_margin              DECIMAL(16,4) NULL COMMENT '毛利(元)',
  extra_item                DECIMAL(16,4) NULL COMMENT '其他项目(万元)',
  profit_dedt               DECIMAL(16,4) NULL COMMENT '扣除非经常性损益后的净利润(万元)',
  op_income                 DECIMAL(16,4) NULL COMMENT '营业收入(万元)',
  daa                       DECIMAL(16,4) NULL COMMENT '折旧和摊销(万元)',
  rd_exp                    DECIMAL(16,4) NULL COMMENT '研发费用(万元)',

  -- 2. Solvency indicators (ratios - no unit conversion needed)
  current_ratio             FLOAT NULL COMMENT '流动比率',
  quick_ratio               FLOAT NULL COMMENT '速动比率',
  cash_ratio                FLOAT NULL COMMENT '现金比率',
  debt_to_assets            FLOAT NULL COMMENT '资产负债率',
  assets_to_eqt             FLOAT NULL COMMENT '权益乘数',
  dp_assets_to_eqt          FLOAT NULL COMMENT '有形资产权益乘数',
  debt_to_eqt               FLOAT NULL COMMENT '产权比率',
  eqt_to_debt               FLOAT NULL COMMENT '净资产负债率倒数',
  eqt_to_interestdebt       FLOAT NULL COMMENT '净资产利息负担率倒数',
  ebit_to_interest          FLOAT NULL COMMENT '利息保障倍数',
  ebitda_to_debt            FLOAT NULL COMMENT 'EBITDA对债务比率',
  debt_to_assets_2          FLOAT NULL COMMENT '资产负债率(备选)',
  assets_to_eqt_2           FLOAT NULL COMMENT '权益乘数(备选)',
  dp_assets_to_eqt_2        FLOAT NULL COMMENT '有形资产权益乘数(备选)',
  tangibleasset_to_debt     FLOAT NULL COMMENT '有形资产债务率',
  tangasset_to_intdebt      FLOAT NULL COMMENT '有形资产利息债务率',
  tangibleasset_to_netdebt  FLOAT NULL COMMENT '有形资产净债务率',

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
  current_exint             DECIMAL(16,4) NULL,
  non_current_exint         DECIMAL(16,4) NULL,
  intrinsicvalue            DECIMAL(16,4) NULL,
  tmv                       DECIMAL(16,4) NULL,
  lmv                       DECIMAL(16,4) NULL,

  -- TTM (Trailing Twelve Months) indicators
  eps_ttm                   FLOAT NULL COMMENT 'TTM每股收益(元)',
  revenue_ps_ttm            FLOAT NULL COMMENT 'TTM每股营收(元)',
  ocfps_ttm                 FLOAT NULL COMMENT 'TTM每股经营现金流(元)',
  cfps_ttm                  FLOAT NULL COMMENT 'TTM每股现金流(元)',
  roe_ttm                   FLOAT NULL COMMENT 'TTM净资产收益率(%)',
  roa_ttm                   FLOAT NULL COMMENT 'TTM总资产报酬率(%)',
  netprofit_margin_ttm      FLOAT NULL COMMENT 'TTM净利率(%)',
  grossprofit_margin_ttm    FLOAT NULL COMMENT 'TTM毛利率(%)',
  revenue_cagr_3y           FLOAT NULL COMMENT '营收三年复合增长率(%)',
  netincome_cagr_3y         FLOAT NULL COMMENT '净利润三年复合增长率(%)',
  roic_ttm                  FLOAT NULL COMMENT 'TTM投资回报率(%)',
  fcf_ttm                   DECIMAL(16,4) NULL COMMENT 'TTM自由现金流(万元)',
  fcf_margin_ttm            FLOAT NULL COMMENT 'TTM自由现金流率(%)',
  debt_to_ebitda_ttm        FLOAT NULL COMMENT 'TTM债务/EBITDA比率',

  PRIMARY KEY (ts_code, report_period),
  INDEX idx_ann_date (ann_date),
  INDEX idx_report_period (report_period),
  INDEX idx_ts_code_ann_date (ts_code, ann_date),
  INDEX idx_ts_code_report_period_ann_date (ts_code, report_period, ann_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;
"""


# === Data source field grouping definitions ===

# Base fields (shared by all data sources)
BASE_COLUMNS = ["ts_code", "ann_date", "report_period", "period", "currency"]

# Income statement fields
INCOME_COLUMNS = [
    "total_revenue", "revenue", "operate_profit", "total_profit", "n_income_attr_p", "basic_eps",
    "total_cogs", "oper_cost", "sell_exp", "admin_exp", "fin_exp", "invest_income", "interest_exp",
    "oper_exp", "ebit", "ebitda", "income_tax", "comshare_payable_dvd", "rd_exp"
]

# Balance sheet fields
BALANCE_COLUMNS = [
    "total_assets", "total_liab", "total_hldr_eqy_inc_min_int", "total_cur_assets",
    "total_cur_liab", "accounts_receiv", "inventories", "acct_payable",
    "fix_assets", "lt_borr", "r_and_d", "goodwill", "intang_assets", "st_borr",
    "total_share", "oth_eqt_tools_p_shr", "total_hldr_eqy_exc_min_int"
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
    "ebitda_margin", "extra_item", "profit_dedt", "op_income", "daa",

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
    "current_exint", "non_current_exint", "intrinsicvalue", "tmv", "lmv",

    # TTM (Trailing Twelve Months) indicators
    "eps_ttm", "revenue_ps_ttm", "ocfps_ttm", "cfps_ttm",
    "roe_ttm", "roa_ttm", "netprofit_margin_ttm", "grossprofit_margin_ttm",
    "revenue_cagr_3y", "netincome_cagr_3y",
    "roic_ttm", "fcf_ttm", "fcf_margin_ttm", "debt_to_ebitda_ttm"
]

# === Data source field configuration ===

# API field name list (all three major financial statements contain these base fields)
# Note: API returns 'end_date' but database stores as 'ann_date'
API_COMMON_FIELDS = ['ts_code', 'ann_date', 'end_date', 'report_type']

# Financial indicators base fields (does not include report_type)
INDICATOR_BASE_FIELDS = ['ts_code', 'ann_date', 'end_date']  # Keep end_date for API call, will be mapped later

# === Merged total field list (used for database operations) ===
ALL_COLUMNS: List[str] = BASE_COLUMNS + INCOME_COLUMNS + BALANCE_COLUMNS + CASHFLOW_COLUMNS + INDICATOR_COLUMNS

# Fields that need conversion from 元 to 万元 for storage
# These are monetary amount fields (not ratios or per-share metrics)
YUAN_TO_WAN_FIELDS = [
    # Income statement - main monetary amounts
    'total_revenue', 'revenue', 'operate_profit', 'total_profit', 'n_income_attr_p',
    'total_cogs', 'oper_cost', 'sell_exp', 'admin_exp', 'fin_exp', 'invest_income',
    'interest_exp', 'oper_exp', 'ebit', 'ebitda', 'income_tax',
    'comshare_payable_dvd', 'rd_exp',

    # Balance sheet - main monetary amounts
    'total_assets', 'total_liab', 'total_cur_assets', 'total_cur_liab', 
    'accounts_receiv', 'inventories', 'acct_payable', 'fix_assets', 
    'lt_borr', 'r_and_d', 'goodwill', 'intang_assets', 'st_borr',
    'total_share', 'oth_eqt_tools_p_shr', 'total_hldr_eqy_exc_min_int', 
    'total_hldr_eqy_inc_min_int',

    # Cash flow statement - main monetary amounts
    'n_cashflow_act', 'n_cashflow_inv_act', 'n_cash_flows_fnc_act',
    'free_cashflow', 'c_pay_acq_const_fiolta', 'c_fr_sale_sg',
    'c_paid_goods_s', 'c_paid_to_for_empl', 'c_paid_for_taxes',
    'n_incr_cash_cash_equ', 'c_disp_withdrwl_invest',
    'c_pay_dist_dpcp_int_exp', 'c_cash_equ_end_period',

    # Financial indicators - monetary amounts (not ratios)
    'extra_item', 'profit_dedt', 'op_income', 'daa', 'gross_margin',

    # Quarterly financial indicators
    'q_opincome', 'q_investincome', 'q_dtprofit',

    # Other monetary amounts
    'profit_prefin_exp', 'non_op_profit', 'fixed_assets',

    # TTM monetary amounts
    'fcf_ttm',

    # Valuation indicators
    'current_exint', 'non_current_exint', 'intrinsicvalue', 'tmv', 'lmv'
]

# Per-share metrics that should remain in 元 (not converted)
PER_SHARE_FIELDS = [
    'eps', 'dt_eps', 'basic_eps', 'q_eps',
    'bps', 'ocfps', 'retainedps', 'cfps', 'ebit_ps', 'fcff_ps', 'fcfe_ps'
]

# TTM indicator fields to add to database schema
TTM_COLUMNS = [
    # TTM basic financial indicators
    'eps_ttm', 'revenue_ps_ttm', 'ocfps_ttm', 'cfps_ttm',
    'roe_ttm', 'roa_ttm', 'netprofit_margin_ttm', 'grossprofit_margin_ttm',

    # TTM growth indicators
    'revenue_cagr_3y', 'netincome_cagr_3y',

    # TTM efficiency and quality indicators
    'roic_ttm', 'fcf_ttm', 'fcf_margin_ttm', 'debt_to_ebitda_ttm'
]


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
    def complete_quarters(ts_code_group):
        ts_code, group = ts_code_group

        # 找到实际存在数据的日期范围
        existing_dates = group['report_date'].dropna().sort_values()

        if len(existing_dates) < 2:
            # 如果数据点太少，无法确定补全范围，直接返回原数据
            group_copy = group.copy()
            group_copy['missing'] = 0  # 标记为非缺失
            group_copy['missing_type'] = 'insufficient_data'
            return group_copy

        min_date = existing_dates.min()
        max_date = existing_dates.max()

        # 生成从最早数据到最晚数据的完整季度末序列
        full_dates = pd.date_range(start=min_date, end=max_date, freq='QE-SEP')
        full_df = pd.DataFrame({'report_date': full_dates})
        full_df['report_period'] = full_df['report_date'].dt.strftime('%Y%m%d')
        full_df['ts_code'] = ts_code  # 添加ts_code

        # 左合并原数据，缺失处NA
        merged = pd.merge(full_df, group, on=['ts_code', 'report_period', 'report_date'], how='left')

        # 分析缺失模式
        missing_mask = merged['n_income_attr_p'].isna()
        merged['missing'] = missing_mask.astype(int)

        # 识别缺失类型
        merged['missing_type'] = 'none'
        merged.loc[missing_mask, 'missing_type'] = 'data_missing'

        # 进一步分类缺失类型
        existing_periods = set(group['report_period'].dropna())
        if existing_periods:
            min_existing_period = min(existing_periods)
            max_existing_period = max(existing_periods)

            # 两头缺失：数据范围外的缺失
            outside_range = (merged['report_period'] < min_existing_period) | (merged['report_period'] > max_existing_period)
            merged.loc[missing_mask & outside_range, 'missing_type'] = 'edge_missing'

            # 中间缺失：数据范围内的缺失
            inside_range = (merged['report_period'] >= min_existing_period) & (merged['report_period'] <= max_existing_period)
            merged.loc[missing_mask & inside_range, 'missing_type'] = 'intermediate_missing'

            # 记录不同类型的缺失
            intermediate_missing = merged[(merged['missing_type'] == 'intermediate_missing')]
            if not intermediate_missing.empty:
                missing_periods = intermediate_missing['report_period'].tolist()
                print(f"⚠️  {ts_code}: 中间数据缺失 {len(missing_periods)} 个季度: {missing_periods}")

        # 统计缺失情况
        missing_stats = merged['missing_type'].value_counts()
        if missing_stats.get('intermediate_missing', 0) > 0:
            print(f"📊 {ts_code}: 缺失统计 - 中间:{missing_stats.get('intermediate_missing', 0)}, 边缘:{missing_stats.get('edge_missing', 0)}, 数据:{missing_stats.get('data_missing', 0)}")

        return merged

    # 使用itertools.groupby来避免pandas groupby的FutureWarning
    import itertools
    df = df.sort_values(['ts_code', 'report_date'])
    groups = []
    for ts_code, group in itertools.groupby(df.iterrows(), key=lambda x: x[1]['ts_code']):
        group_df = pd.DataFrame([row[1] for row in group])
        groups.append((ts_code, group_df))

    completed_groups = [complete_quarters((ts_code, group)) for ts_code, group in groups]
    df = pd.concat(completed_groups, ignore_index=True)

    # 智能填充NA数据，根据缺失类型采用不同策略
    print("🔄 开始智能数据填充...")

    # 1. 流数据（flow data）：收入、成本、现金流等
    # 中间缺失使用插值，两头缺失保持为0（表示该时期没有数据）
    flow_cols = ['n_income_attr_p', 'total_revenue', 'im_net_cashflow_oper_act']
    optional_flow_cols = ['total_cogs', 'oper_cost']
    for col in optional_flow_cols:
        if col in df.columns:
            flow_cols.append(col)

    for col in flow_cols:
        if col in df.columns:
            # 对于中间缺失，使用线性插值
            intermediate_mask = df['missing_type'] == 'intermediate_missing'
            if intermediate_mask.any():
                # 对每个股票分别进行插值
                df[col] = df.groupby('ts_code')[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))

            # 对于两头缺失和数据缺失，填充为0（表示该时期没有发生）
            edge_data_mask = (df['missing_type'] == 'edge_missing') | (df['missing_type'] == 'data_missing')
            df.loc[edge_data_mask, col] = df.loc[edge_data_mask, col].fillna(0)

    # 2. 存量数据（stock data）：资产、负债、股权等
    # 使用前向填充，然后后向填充，确保连续性
    stock_cols = ['total_hldr_eqy_exc_min_int', 'total_assets', 'total_share']
    for col in stock_cols:
        if col in df.columns:
            # 使用transform来避免groupby的FutureWarning
            df[col] = df.groupby('ts_code')[col].transform(lambda x: x.ffill().bfill())

    # 3. 特殊处理：某些关键指标如果仍然缺失，使用行业平均或其他方法
    # 这里可以添加更复杂的填充逻辑

    # 统计填充结果
    remaining_na = df[flow_cols + stock_cols].isna().sum().sum()
    if remaining_na > 0:
        print(f"⚠️  仍有 {remaining_na} 个值未填充")
    else:
        print("✅ 数据填充完成，无剩余缺失值")

    quarterly_columns = ['n_income_attr_p', 'total_revenue', 'im_net_cashflow_oper_act']
    # Add optional quarterly columns that might not exist in all datasets
    optional_quarterly_cols = ['total_cogs', 'oper_cost']
    for col in optional_quarterly_cols:
        if col in df.columns:
            quarterly_columns.append(col)

    # 使用itertools.groupby来避免pandas groupby的FutureWarning
    df = df.sort_values(['ts_code', 'report_period'])
    groups = []
    for ts_code, group in itertools.groupby(df.iterrows(), key=lambda x: x[1]['ts_code']):
        group_df = pd.DataFrame([row[1] for row in group])
        groups.append((ts_code, group_df))

    processed_groups = [calculate_quarterly_values(group, quarterly_columns) for ts_code, group in groups]
    df = pd.concat(processed_groups, ignore_index=True)

    # Sort by ts_code and report_period
    df = df.sort_values(['ts_code', 'report_period'])

    # Calculate rolling TTM sums for quarterly values
    ttm_columns = {col: 'ttm_' + col for col in quarterly_columns}
    for q_col, ttm_col in ttm_columns.items():
        df[ttm_col] = df.groupby('ts_code')['q_' + q_col].rolling(window=4, min_periods=4).sum().reset_index(level=0, drop=True)

    # Drop rows where TTM is NaN (insufficient history)
    #df = df.dropna(subset=list(ttm_columns.values()))

    # Calculate TTM gross (only if oper_cost data is available)
    if 'ttm_oper_cost' in df.columns:
        df['ttm_gross'] = df['ttm_total_revenue'] - df['ttm_oper_cost']
    else:
        df['ttm_gross'] = df['ttm_total_revenue']  # Fallback if no cost data

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
    # Gross margin only if cost data is available
    if 'ttm_oper_cost' in df.columns:
        df['grossprofit_margin_ttm'] = np.where(df['ttm_total_revenue'] > 0,
                                                (df['ttm_gross'] / df['ttm_total_revenue']) * 100, 0)
    else:
        df['grossprofit_margin_ttm'] = np.nan

    # CAGR (3-year, same quarter) with special handling for negative values
    df['revenue_3y_ago'] = df.groupby('ts_code')['total_revenue'].shift(12)
    df['ni_3y_ago'] = df.groupby('ts_code')['n_income_attr_p'].shift(12)

    # Revenue CAGR calculation with negative value handling
    df['revenue_cagr_3y'] = np.nan

    # Both positive (normal CAGR)
    mask_both_positive = (df['revenue_3y_ago'] > 0) & (df['total_revenue'] > 0)
    df.loc[mask_both_positive, 'revenue_cagr_3y'] = (
        (df.loc[mask_both_positive, 'total_revenue'] / df.loc[mask_both_positive, 'revenue_3y_ago']) ** (1/3) - 1
    ) * 100

    # Net Income CAGR calculation with similar logic
    df['netincome_cagr_3y'] = np.nan

    # Both positive (normal CAGR)
    mask_both_positive_ni = (df['ni_3y_ago'] > 0) & (df['n_income_attr_p'] > 0)
    df.loc[mask_both_positive_ni, 'netincome_cagr_3y'] = (
        (df.loc[mask_both_positive_ni, 'n_income_attr_p'] / df.loc[mask_both_positive_ni, 'ni_3y_ago']) ** (1/3) - 1
    ) * 100

    # FCF TTM (Free Cash Flow) - approximation using available data
    # FCF = Operating Cash Flow - CapEx
    if 'n_cashflow_act' in df.columns and 'c_pay_acq_const_fiolta' in df.columns:
        df['fcf_ttm'] = df['n_cashflow_act'] - df['c_pay_acq_const_fiolta'].fillna(0)
    elif 'n_cashflow_act' in df.columns:
        df['fcf_ttm'] = df['n_cashflow_act'] * 0.8  # Rough approximation
    else:
        df['fcf_ttm'] = np.nan

    # FCF Margin TTM
    df['fcf_margin_ttm'] = np.where(df['ttm_total_revenue'] > 0,
                                    (df['fcf_ttm'] / df['ttm_total_revenue']) * 100, np.nan)

    # Debt to EBITDA TTM ratio
    if 'total_liab' in df.columns and 'ebitda' in df.columns:
        df['debt_to_ebitda_ttm'] = np.where(df['ebitda'] > 0, df['total_liab'] / df['ebitda'], np.nan)
    else:
        df['debt_to_ebitda_ttm'] = np.nan

    # Round results
    round_cols = ['eps_ttm', 'revenue_ps_ttm', 'ocfps_ttm', 'cfps_ttm', 'roe_ttm', 'roa_ttm',
                  'netprofit_margin_ttm', 'grossprofit_margin_ttm', 'revenue_cagr_3y', 'netincome_cagr_3y',
                  'fcf_margin_ttm', 'debt_to_ebitda_ttm']
    df[round_cols] = df[round_cols].round(4)

    # Remove filled rows (missing=1) after calculations are complete
    if 'missing' in df.columns:
        original_count = len(df)
        df = df[df['missing'] != 1].copy()
        removed_count = original_count - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} filled rows after TTM/CAGR calculations")
        df = df.drop(columns=['missing'])

    return df


def convert_wan_to_yuan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert stored 万元 values back to 元 for API responses
    This ensures API consumers get data in the expected 元 format

    Args:
        df: DataFrame with values stored in 万元

    Returns:
        DataFrame with monetary values converted back to 元
    """
    if df.empty:
        return df

    df_copy = df.copy()

    # Convert 万元 back to 元 for monetary amount fields
    for col in YUAN_TO_WAN_FIELDS:
        if col in df_copy.columns and df_copy[col].notna().any():
            # Convert 万元 to 元 (multiply by 10,000)
            df_copy[col] = df_copy[col] * 10000.0

    return df_copy


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
        # String columns (excluding date columns that will be converted to DATE type)
        string_cols = ["ts_code", "period", "currency"]
        for col in string_cols:
            if col in out.columns:
                out[col] = out[col].astype(str).replace('nan', None).replace('None', None)

        # Convert date columns from string to DATE objects for efficient storage and queries
        # This avoids SQL-level date conversion and improves insertion performance
        # Convert report_period from '2024-03-31' format to DATE object
        if 'report_period' in out.columns:
            out['report_period'] = pd.to_datetime(out['report_period'], format='%Y-%m-%d', errors='coerce').dt.date

        # Convert ann_date from '20240331' format to DATE object
        if 'ann_date' in out.columns:
            out['ann_date'] = pd.to_datetime(out['ann_date'], format='%Y%m%d', errors='coerce').dt.date

        # Numeric columns - convert to float first, then handle None values
        # Exclude string columns and date columns (report_period, ann_date)
        date_cols = ["report_period", "ann_date"]
        numeric_cols = [col for col in ALL_COLUMNS if col not in string_cols and col not in date_cols]
        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        # Convert monetary amounts from 元 to 万元 for storage
        # This reduces storage space and prevents DECIMAL overflow
        for col in YUAN_TO_WAN_FIELDS:
            if col in out.columns and out[col].notna().any():
                # Convert 元 to 万元 (divide by 10,000)
                original_values = out[col].copy()
                out[col] = out[col] / 10000.0

                # Log conversion for large values
                large_conversions = original_values.abs() > 1000000000  # > 10亿
                if large_conversions.any():
                    max_original = original_values[large_conversions].max()
                    converted = out[col][large_conversions].max()
                    print(f"Converted {col}: {max_original:.0f}元 → {converted:.4f}万元")

                print(f"Converted {col} from 元 to 万元 for storage")

        # Validate and clamp numeric values to prevent DECIMAL overflow
        # After conversion to 万元, the limits are much more generous:
        # - DECIMAL(16,4): max ~999,999,999,999万元 (999万亿), min ~-999,999,999,999万元
        # - DECIMAL(18,4): max ~99,999,999,999,999万元 (99万亿), min ~-99,999,999,999,999万元
        # - DECIMAL(22,4): max ~99,999,999,999,999,999万元 (99万亿), min ~-99,999,999,999,999,999万元
        decimal_limits = {
            # After 元→万元 conversion, limits are very generous for most financial data
            'total_revenue': (16, 4),           # DECIMAL(16,4) - up to ~999万亿万元
            'revenue': (16, 4),                 # DECIMAL(16,4)
            'operate_profit': (16, 4),          # DECIMAL(16,4)
            'total_profit': (16, 4),            # DECIMAL(16,4)
            'n_income_attr_p': (16, 4),         # DECIMAL(16,4)
            'basic_eps': None,                  # FLOAT - per-share metrics remain in 元
            'total_cogs': (16, 4),              # DECIMAL(16,4)
            'oper_cost': (16, 4),               # DECIMAL(16,4)
            'sell_exp': (16, 4),                # DECIMAL(16,4)
            'admin_exp': (16, 4),               # DECIMAL(16,4)
            'fin_exp': (16, 4),                 # DECIMAL(16,4)
            'invest_income': (16, 4),           # DECIMAL(16,4)
            'interest_exp': (16, 4),            # DECIMAL(16,4)
            'oper_exp': (16, 4),                # DECIMAL(16,4)
            'ebit': (16, 4),                    # DECIMAL(16,4)
            'ebitda': (16, 4),                  # DECIMAL(16,4)
            'income_tax': (16, 4),              # DECIMAL(16,4)
            'comshare_payable_dvd': (16, 4),    # DECIMAL(16,4)

            # Balance sheet fields - very generous limits after conversion
            'total_assets': (16, 4),            # DECIMAL(16,4)
            'total_liab': (16, 4),              # DECIMAL(16,4)
            'total_hldr_eqy_inc_min_int': (16, 4), # DECIMAL(16,4)
            'total_cur_assets': (16, 4),        # DECIMAL(18,4)
            'total_cur_liab': (16, 4),          # DECIMAL(18,4)
            'accounts_receiv': (16, 4),         # DECIMAL(16,4)
            'inventories': (16, 4),             # DECIMAL(16,4)
            'acct_payable': (16, 4),            # DECIMAL(16,4)
            'fix_assets': (16, 4),              # DECIMAL(16,4)
            'lt_borr': (16, 4),                 # DECIMAL(16,4)
            'r_and_d': (16, 4),                 # DECIMAL(16,4)
            'goodwill': (16, 4),                # DECIMAL(16,4)
            'intang_assets': (16, 4),           # DECIMAL(16,4)
            'st_borr': (16, 4),                 # DECIMAL(16,4)
            'total_share': (16, 4),             # DECIMAL(16,4)
            'oth_eqt_tools_p_shr': (16, 4),     # DECIMAL(16,4)

            # Cash flow fields
            'n_cashflow_act': (16, 4),          # DECIMAL(16,4)
            'n_cashflow_inv_act': (16, 4),      # DECIMAL(16,4)
            'n_cash_flows_fnc_act': (16, 4),    # DECIMAL(16,4)
            'free_cashflow': (16, 4),           # DECIMAL(16,4)
            'c_pay_acq_const_fiolta': (16, 4),  # DECIMAL(16,4)
            'c_fr_sale_sg': (16, 4),            # DECIMAL(16,4)
            'c_paid_goods_s': (16, 4),          # DECIMAL(16,4)
            'c_paid_to_for_empl': (16, 4),      # DECIMAL(16,4)
            'c_paid_for_taxes': (16, 4),        # DECIMAL(16,4)
            'n_incr_cash_cash_equ': (16, 4),    # DECIMAL(16,4)
            'c_disp_withdrwl_invest': (16, 4),  # DECIMAL(16,4)
            'c_pay_dist_dpcp_int_exp': (16, 4), # DECIMAL(16,4)
            'c_cash_equ_end_period': (16, 4),   # DECIMAL(16,4)

            # Financial indicator fields (ratios/per-share metrics - no conversion)
            'eps': None,                        # FLOAT - per-share (元)
            'dt_eps': None,                     # FLOAT - per-share (元)
            'gross_margin': None,               # FLOAT - ratio (%)
            'netprofit_margin': None,           # FLOAT - ratio (%)
            'grossprofit_margin': None,         # FLOAT - ratio (%)
            'ebitda_margin': None,              # FLOAT - ratio (%)
            'extra_item': (16, 4),              # DECIMAL(16,4) - monetary amount
            'profit_dedt': (16, 4),             # DECIMAL(16,4) - monetary amount
            'op_income': (16, 4),               # DECIMAL(16,4) - monetary amount
            'daa': (16, 4),                     # DECIMAL(16,4) - monetary amount
            'rd_exp': (16, 4),                  # DECIMAL(16,4) - monetary amount

            # Most other indicator fields are FLOAT (ratios) or smaller ranges
            # For safety, we'll use conservative limits for large values
        }

        # Add conservative default limits for any DECIMAL fields not explicitly defined
        # This handles cases where the database schema might differ from our assumptions
        default_decimal_limit = (16, 4)  # Conservative default: DECIMAL(16,4)

        # Check for any numeric columns that might be DECIMAL but aren't in our limits dict
        for col in numeric_cols:
            if col not in out.columns:
                continue

            if col not in decimal_limits:
                # Check if column might contain large values that could overflow
                col_values = pd.to_numeric(out[col], errors='coerce')
                if col_values.notna().any():
                    max_val = col_values.abs().max()  # Check absolute value
                    if max_val > 1000000000:  # If greater than 1 billion
                        # Apply conservative limit to prevent overflow
                        precision, scale = default_decimal_limit
                        max_allowed = 10 ** (precision - scale) - (10 ** (-scale))
                        min_allowed = -max_allowed

                        if max_val > max_allowed:
                            print(f"Warning: {col} has large value {max_val}, applying conservative limit")
                            out[col] = out[col].clip(lower=min_allowed, upper=max_allowed)

        # Apply decimal limits to prevent overflow
        for col in numeric_cols:
            if col in out.columns and decimal_limits.get(col) is not None:
                precision, scale = decimal_limits[col]
                max_value = 10 ** (precision - scale) - (10 ** (-scale))
                min_value = -max_value

                # Debug: Check for values that exceed limits before clamping
                if out[col].notna().any():
                    original_series = pd.to_numeric(out[col], errors='coerce')
                    exceeding_max = original_series > max_value
                    exceeding_min = original_series < min_value

                    if exceeding_max.any():
                        max_val = original_series[exceeding_max].max()
                        print(f"Warning: {col} has value {max_val} exceeding max {max_value}")
                    if exceeding_min.any():
                        min_val = original_series[exceeding_min].min()
                        print(f"Warning: {col} has value {min_val} below min {min_value}")

                # Clamp values to valid range
                out[col] = out[col].clip(lower=min_value, upper=max_value)

                # Log clamping results
                if out[col].notna().any():
                    final_max = pd.to_numeric(out[col], errors='coerce').max()
                    final_min = pd.to_numeric(out[col], errors='coerce').min()
                    if final_max == max_value:
                        print(f"Info: {col} clamped to max value {max_value}")
                    if final_min == min_value:
                        print(f"Info: {col} clamped to min value {min_value}")

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
            period=report_period,
            fields=','.join(income_fields)
        )

        # 2. Get balance sheet data (with retry mechanism)
        balance_fields = API_COMMON_FIELDS + BALANCE_COLUMNS
        balance_df = call_tushare_api_with_retry(
            pro.balancesheet_vip,
            period=report_period,
            fields=','.join(balance_fields)
        )

        # 3. Get cash flow statement data (with retry mechanism)
        cashflow_fields = API_COMMON_FIELDS + CASHFLOW_COLUMNS
        cashflow_df = call_tushare_api_with_retry(
            pro.cashflow_vip,
            period=report_period,
            fields=','.join(cashflow_fields)
        )

        # 4. Get financial indicators data (with retry mechanism)
        indicator_fields = INDICATOR_BASE_FIELDS + INDICATOR_COLUMNS
        indicator_df = call_tushare_api_with_retry(
            pro.fina_indicator_vip,
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
            if len(df) > 0:
                # Remove duplicates within each data source before merging
                # Keep the last record (potentially more updated/corrected data)
                initial_len = len(df)
                df = df.drop_duplicates(subset=['ts_code', 'ann_date', 'end_date'], keep='last')
                if len(df) < initial_len:
                    print(f"Removed {initial_len - len(df)} duplicates from {name} (kept latest)")

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

                    merged_df = merged_df.merge(
                        df,
                        on=API_COMMON_FIELDS,
                        how='outer'
                    )
                    after_count = len(merged_df)
                    print(f"After: {after_count} records (+{after_count - before_count})")

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
                print(f"Base data: {len(base_keys)} unique combinations, {before_count} total records")
                print(f"Indicators: {len(indicator_keys)} unique combinations, {len(indicator_df)} total records")

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

        # Add unified fields
        try:
            merged_df['ts_code'] = merged_df['ts_code']

            # Keep the original ann_date from API - don't override with end_date
            # ann_date represents when the financial report was actually announced
            # end_date represents the end of the reporting period
            if 'ann_date' not in merged_df.columns or merged_df['ann_date'].isna().all():
                # Fallback to end_date only if ann_date is missing
                merged_df['ann_date'] = merged_df['end_date']
                print(f"Warning: Using end_date as ann_date fallback for period {report_period}")
            else:
                # Validate ann_date makes sense (should be after end_date)
                sample_ann_date = merged_df['ann_date'].iloc[0]
                sample_end_date = merged_df['end_date'].iloc[0]
                print(f"Using API ann_date: {sample_ann_date} (end_date: {sample_end_date}) for period {report_period}")

            # Convert report_period to DATE object directly from end_date
            # This creates a proper DATE object for the reporting period end date
            merged_df['report_period'] = pd.to_datetime(merged_df['end_date'], format='%Y%m%d').dt.date
            merged_df['period'] = 'annual' if report_period.endswith('1231') else 'quarter'
            merged_df['currency'] = 'CNY'  # A-share default currency is CNY

            # Remove API-specific fields that don't exist in database
            if 'end_date' in merged_df.columns:
                merged_df = merged_df.drop('end_date', axis=1)

            # Remove duplicates based on primary key (ts_code, report_period)
            # Keep the last record (potentially more updated/corrected data)
            initial_count = len(merged_df)

            # Debug: Check for duplicates before removal
            duplicate_check = merged_df.groupby(['ts_code', 'report_period']).size()
            duplicates_found = duplicate_check[duplicate_check > 1]
            if len(duplicates_found) > 0:
                print(f"Found {len(duplicates_found)} duplicate groups before removal:")
                for (ts_code, report_period), count in duplicates_found.items():
                    print(f"  {ts_code} {report_period}: {count} duplicates")

            merged_df = merged_df.drop_duplicates(subset=['ts_code', 'report_period'], keep='last')
            final_count = len(merged_df)

            if initial_count != final_count:
                print(f"Removed {initial_count - final_count} duplicate records, kept {final_count} unique records (latest)")
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
                if c not in ("ts_code", "report_period", "ann_date")  # Exclude primary key and date fields from updates
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
    - Date fields: DATE type (report_period, ann_date) for efficient date operations and storage

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

    # Collect all data first for TTM calculations
    all_data_frames = []
    total_raw_records = 0

    for i, report_period in enumerate(periods):
        print(f"\nProcessing period {i+1}/{len(periods)}: {report_period}")

        # Get data for single period
        df = _fetch_single_period_data(report_period)

        if df.empty:
            print(f"No data retrieved for period {report_period}, skipping")
            continue

        # Data normalization
        df = _coerce_schema(df)
        all_data_frames.append(df)
        total_raw_records += len(df)

        print(f"Retrieved {len(df)} financial profile records for period {report_period}")

        # Add delay to avoid API limits (already added in _fetch_single_period_data)
        if i < len(periods) - 1:  # Not the last period, add delay
            time.sleep(0.5)

    if not all_data_frames:
        print("No data retrieved for any period")
        return

    # Combine all data for TTM calculations
    combined_df = pd.concat(all_data_frames, ignore_index=True)
    print(f"\nCombined {len(all_data_frames)} periods into {len(combined_df)} total records")

    # Calculate TTM indicators
    print("Calculating TTM (Trailing Twelve Months) indicators...")
    combined_df = calculate_ttm_indicators(combined_df)
    print(f"TTM calculation completed, {len(combined_df)} records after TTM processing")

    # Upsert to database in batches
    total_written = _upsert_batch(engine, combined_df, chunksize=chunksize)

    print(f"\nUpdate completed:")
    print(f"- Processed {len(periods)} periods")
    print(f"- Retrieved {total_raw_records} raw records")
    print(f"- Final records after TTM calculation: {len(combined_df)}")
    print(f"- Total records written to database: {total_written}")


if __name__ == "__main__":
    fire.Fire(update_a_stock_financial_profile)
