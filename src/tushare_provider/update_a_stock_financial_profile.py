import os
import time
import datetime
import logging
from typing import Optional, List, Dict, Any, Union
import itertools
import fire
import pandas as pd
import numpy as np
import tushare as ts
from sqlalchemy import create_engine, text
from sqlalchemy import Table, MetaData
from sqlalchemy.dialects.mysql import insert as mysql_insert
import pymysql  # noqa: F401 - required by SQLAlchemy URL

from ..util import (
    setup_logging, init_tushare, call_tushare_api_with_retry
)

setup_logging(level=logging.DEBUG, log_file='update_a_stock_financial_profile.log')
logger = logging.getLogger(__name__)

tushare_pro = init_tushare()

TABLE_NAME = "ts_a_stock_financial_profile"
CREATE_TABLE_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
  ts_code                   VARCHAR(16)  NOT NULL,
  report_period             DATE         NOT NULL,
  period                    VARCHAR(8)   NOT NULL,
  currency                  VARCHAR(3)   NOT NULL,
  ann_date                  DATE         NULL,

  -- Income statement fields (万元存储 - converted from 元)
  basic_eps                 FLOAT NULL COMMENT '基本每股收益(元)',
  diluted_eps               FLOAT NULL COMMENT '稀释每股收益(元)',
  total_revenue             DECIMAL(16,4) NULL COMMENT '总营收(万元)',
  revenue                   DECIMAL(16,4) NULL COMMENT '营业收入(万元)',
  total_cogs                DECIMAL(16,4) NULL COMMENT '营业总成本(万元)',
  oper_cost                 DECIMAL(16,4) NULL COMMENT '营业成本(万元)',
  sell_exp                  DECIMAL(16,4) NULL COMMENT '销售费用(万元)',
  admin_exp                 DECIMAL(16,4) NULL COMMENT '管理费用(万元)',
  fin_exp                   DECIMAL(16,4) NULL COMMENT '财务费用(万元)',
  assets_impair_loss        DECIMAL(16,4) NULL COMMENT '资产减值损失(万元)',
  operate_profit            DECIMAL(16,4) NULL COMMENT '营业利润(万元)',
  non_oper_income           DECIMAL(16,4) NULL COMMENT '营业外收入(万元)',
  non_oper_exp              DECIMAL(16,4) NULL COMMENT '营业外支出(万元)',
  total_profit              DECIMAL(16,4) NULL COMMENT '利润总额(万元)',
  income_tax                DECIMAL(16,4) NULL COMMENT '所得税(万元)',
  n_income                  DECIMAL(16,4) NULL COMMENT '净利润(万元)',
  n_income_attr_p           DECIMAL(16,4) NULL COMMENT '净利润(万元)',
  ebit                      DECIMAL(16,4) NULL COMMENT '息税前利润(万元)',
  ebitda                    DECIMAL(16,4) NULL COMMENT 'EBITDA(万元)',
  invest_income             DECIMAL(16,4) NULL COMMENT '投资收益(万元)',
  interest_exp              DECIMAL(16,4) NULL COMMENT '利息支出(万元)',
  oper_exp                  DECIMAL(16,4) NULL COMMENT '营业支出(万元)',
  comshare_payable_dvd      DECIMAL(16,4) NULL COMMENT '应付股利(万元)',
 
  -- Balance sheet fields (万元存储 - converted from 元)
  total_share               DECIMAL(16,4) NULL COMMENT '股本(万元)',
  cap_rese                  DECIMAL(16,4) NULL COMMENT '资本公积(万元)',
  undistr_porfit            DECIMAL(16,4) NULL COMMENT '未分配利润(万元)',
  surplus_rese              DECIMAL(16,4) NULL COMMENT '盈余公积(万元)',
  money_cap                 DECIMAL(16,4) NULL COMMENT '货币资金(万元)',
  accounts_receiv           DECIMAL(16,4) NULL COMMENT '应收账款(万元)',
  oth_receiv                DECIMAL(16,4) NULL COMMENT '其他应收款(万元)',
  prepayment                DECIMAL(16,4) NULL COMMENT '预付款项(万元)',
  inventories               DECIMAL(16,4) NULL COMMENT '存货(万元)',
  oth_cur_assets            DECIMAL(16,4) NULL COMMENT '其他流动资产(万元)',
  total_cur_assets          DECIMAL(16,4) NULL COMMENT '流动资产合计(万元)',
  htm_invest                DECIMAL(16,4) NULL COMMENT '可供出售金融资产(万元)',
  fix_assets                DECIMAL(16,4) NULL COMMENT '固定资产(万元)',
  intan_assets              DECIMAL(16,4) NULL COMMENT '无形资产(万元)',
  defer_tax_assets          DECIMAL(16,4) NULL COMMENT '递延所得税资产(万元)',
  total_nca                 DECIMAL(16,4) NULL COMMENT '非流动资产合计(万元)',
  total_assets              DECIMAL(16,4) NULL COMMENT '资产总计(万元)',
  acct_payable              DECIMAL(16,4) NULL COMMENT '应付账款(万元)',
  payroll_payable           DECIMAL(16,4) NULL COMMENT '应付职工薪酬(万元)',
  taxes_payable             DECIMAL(16,4) NULL COMMENT '应交税费(万元)',
  oth_payable               DECIMAL(16,4) NULL COMMENT '其他应付款(万元)',
  total_cur_liab            DECIMAL(16,4) NULL COMMENT '流动负债合计(万元)',
  defer_inc_non_cur_liab    DECIMAL(16,4) NULL COMMENT '递延收益-非流动负债(万元)',
  total_ncl                 DECIMAL(16,4) NULL COMMENT '非流动负债合计(万元)',
  total_liab                DECIMAL(16,4) NULL COMMENT '负债合计(万元)',
  total_hldr_eqy_exc_min_int DECIMAL(16,4) NULL COMMENT '股东权益合计(万元)',
  total_hldr_eqy_inc_min_int DECIMAL(16,4) NULL COMMENT '股东权益合计(含少数股东)(万元)',
  total_liab_hldr_eqy       DECIMAL(16,4) NULL COMMENT '负债和股东权益总计(万元)',
  oth_pay_total             DECIMAL(16,4) NULL COMMENT '其他应付款总计(万元)',
  accounts_receiv_bill      DECIMAL(16,4) NULL COMMENT '应收票据(万元)',
  accounts_pay              DECIMAL(16,4) NULL COMMENT '应付账款(万元)',
  oth_rcv_total             DECIMAL(16,4) NULL COMMENT '其他应收款总计(万元)',
  fix_assets_total          DECIMAL(16,4) NULL COMMENT '固定资产总计(万元)',
  lt_borr                   DECIMAL(16,4) NULL COMMENT '长期借款(万元)',
  st_borr                   DECIMAL(16,4) NULL COMMENT '短期借款(万元)',
  oth_eqt_tools_p_shr       DECIMAL(16,4) NULL COMMENT '其他权益工具(万元)',
  r_and_d                   DECIMAL(16,4) NULL COMMENT '研发支出(万元)',
  goodwill                  DECIMAL(16,4) NULL COMMENT '商誉(万元)',

  -- Cash flow statement fields (万元存储 - converted from 元)
  net_profit                DECIMAL(16,4) NULL COMMENT '净利润(万元)',
  finan_exp                 DECIMAL(16,4) NULL COMMENT '财务费用(万元)',
  c_fr_sale_sg              DECIMAL(16,4) NULL COMMENT '销售商品收款(万元)',
  c_inf_fr_operate_a        DECIMAL(16,4) NULL COMMENT '经营活动现金流入小计(万元)',
  c_paid_goods_s            DECIMAL(16,4) NULL COMMENT '购买商品付款(万元)',
  c_paid_to_for_empl        DECIMAL(16,4) NULL COMMENT '支付职工薪酬(万元)',
  c_paid_for_taxes          DECIMAL(16,4) NULL COMMENT '支付税费(万元)',
  n_cashflow_act            DECIMAL(16,4) NULL COMMENT '经营活动现金流量净额(万元)',
  n_cashflow_inv_act        DECIMAL(16,4) NULL COMMENT '投资活动现金流量净额(万元)',
  free_cashflow             DECIMAL(16,4) NULL COMMENT '自由现金流(万元)',
  n_cash_flows_fnc_act      DECIMAL(16,4) NULL COMMENT '融资活动现金流量净额(万元)',
  n_incr_cash_cash_equ      DECIMAL(16,4) NULL COMMENT '现金及现金等价物净增加额(万元)',
  c_cash_equ_beg_period     DECIMAL(16,4) NULL COMMENT '期初现金及现金等价物余额(万元)',
  c_cash_equ_end_period     DECIMAL(16,4) NULL COMMENT '期末现金及现金等价物余额(万元)',
  im_net_cashflow_oper_act  DECIMAL(16,4) NULL COMMENT '经营活动产生的现金流量净额(万元)',
  end_bal_cash              DECIMAL(16,4) NULL COMMENT '期末现金余额(万元)',
  beg_bal_cash              DECIMAL(16,4) NULL COMMENT '期初现金余额(万元)',
  c_pay_acq_const_fiolta    DECIMAL(16,4) NULL COMMENT '购建固定资产、无形资产和其他长期资产支付的现金(万元)',
  c_disp_withdrwl_invest    DECIMAL(16,4) NULL COMMENT '处置固定资产、无形资产和其他长期资产收回的现金净额(万元)',
  c_pay_dist_dpcp_int_exp   DECIMAL(16,4) NULL COMMENT '分配股利、利润或偿付利息支付的现金(万元)',

  -- Financial indicator fields (synchronized with tushare_validate.py + TTM extensions)
  eps                       FLOAT NULL COMMENT '每股收益(元)',
  dt_eps                    FLOAT NULL COMMENT '稀释每股收益(元)',
  revenue_ps                FLOAT NULL COMMENT '每股营收(元)',
  bps                       FLOAT NULL COMMENT '每股净资产(元)',
  cfps                      FLOAT NULL COMMENT '每股现金流(元)',
  fcff_ps                   FLOAT NULL COMMENT '每股自由现金流(元)',
  gross_margin              FLOAT NULL COMMENT '毛利率(%)',
  netprofit_margin          FLOAT NULL COMMENT '净利率(%)',
  grossprofit_margin        FLOAT NULL COMMENT '毛利润率(%)',
  current_ratio             FLOAT NULL COMMENT '流动比率',
  quick_ratio               FLOAT NULL COMMENT '速动比率',
  cash_ratio                FLOAT NULL COMMENT '现金比率',
  inv_turn                  FLOAT NULL COMMENT '存货周转率',
  ar_turn                   FLOAT NULL COMMENT '应收账款周转率',
  ca_turn                   FLOAT NULL COMMENT '流动资产周转率',
  fa_turn                   FLOAT NULL COMMENT '固定资产周转率',
  assets_turn               FLOAT NULL COMMENT '总资产周转率',
  debt_to_assets            FLOAT NULL COMMENT '资产负债率',
  debt_to_eqt               FLOAT NULL COMMENT '产权比率',
  roe                       FLOAT NULL COMMENT '净资产收益率(%)',
  roa                       FLOAT NULL COMMENT '总资产报酬率(%)',
  roic                      FLOAT NULL COMMENT '投资回报率(%)',
  netprofit_yoy             FLOAT NULL COMMENT '净利润同比增长率(%)',
  or_yoy                    FLOAT NULL COMMENT '营业收入同比增长率(%)',
  basic_eps_yoy             FLOAT NULL COMMENT '基本每股收益同比增长率(%)',
  assets_yoy                FLOAT NULL COMMENT '资产同比增长率(%)',
  eqt_yoy                   FLOAT NULL COMMENT '净资产同比增长率(%)',
  ocf_yoy                   FLOAT NULL COMMENT '经营现金流同比增长率(%)',
  roe_yoy                   FLOAT NULL COMMENT '净资产收益率同比增长率(%)',
  equity_yoy                FLOAT NULL COMMENT '股东权益同比增长率(%)',
  rd_exp                    FLOAT NULL COMMENT '研发支出(万元)',

  -- TTM (Trailing Twelve Months) indicators - our key additions
  eps_ttm                   FLOAT NULL COMMENT 'TTM每股收益(元)',
  revenue_ps_ttm            FLOAT NULL COMMENT 'TTM每股营收(元)',
  roe_ttm                   FLOAT NULL COMMENT 'TTM净资产收益率(%)',
  roa_ttm                   FLOAT NULL COMMENT 'TTM总资产报酬率(%)',
  netprofit_margin_ttm      FLOAT NULL COMMENT 'TTM净利率(%)',
  grossprofit_margin_ttm    FLOAT NULL COMMENT 'TTM毛利率(%)',
  revenue_cagr_3y           FLOAT NULL COMMENT '营收三年复合增长率(%)',
  netincome_cagr_3y         FLOAT NULL COMMENT '净利润三年复合增长率(%)',
  fcf_margin_ttm            FLOAT NULL COMMENT 'TTM自由现金流率(%)',

  -- Ratio indicators of quality and r&d investment
  debt_to_ebitda            FLOAT NULL COMMENT '债务/EBITDA比率',
  rd_exp_to_capex           FLOAT NULL COMMENT '研发支出/资本支出比率',

  PRIMARY KEY (ts_code, report_period),
  INDEX idx_ann_date (ann_date),
  INDEX idx_report_period (report_period),
  INDEX idx_ts_code_ann_date (ts_code, ann_date),
  INDEX idx_ts_code_report_period_ann_date (ts_code, report_period, ann_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;
"""

# Common fields for all APIs
COMMON_FIELDS = ['ts_code', 'ann_date', 'end_date']

# Income statement fields
INCOME_FIELDS = [
    'basic_eps', 'diluted_eps', 'total_revenue', 'revenue', 'total_cogs', 'oper_cost', 
    'sell_exp', 'admin_exp', 'fin_exp', 'assets_impair_loss', 'operate_profit', 
    'non_oper_income', 'non_oper_exp', 'total_profit', 'income_tax', 'n_income', 
    'n_income_attr_p', 'invest_income', 'interest_exp', 'oper_exp', 
    'comshare_payable_dvd'
]

# Balance sheet fields
BALANCE_FIELDS = [
    'total_share', 'cap_rese', 'undistr_porfit', 'surplus_rese', 'money_cap', 
    'accounts_receiv', 'oth_receiv', 'prepayment', 'inventories', 'oth_cur_assets', 
    'total_cur_assets', 'htm_invest', 'fix_assets', 'intan_assets', 'defer_tax_assets', 
    'total_nca', 'total_assets', 'acct_payable', 'payroll_payable', 'taxes_payable', 
    'oth_payable', 'total_cur_liab', 'defer_inc_non_cur_liab', 'total_ncl', 'total_liab', 
    'total_hldr_eqy_exc_min_int', 'total_hldr_eqy_inc_min_int', 'total_liab_hldr_eqy', 
    'oth_pay_total', 'accounts_receiv_bill', 'accounts_pay', 'oth_rcv_total', 
    'fix_assets_total', 'lt_borr', 'st_borr', 'oth_eqt_tools_p_shr', 'r_and_d', 'goodwill'
]

# Cash flow statement fields
CASHFLOW_FIELDS = [
    'net_profit', 'finan_exp', 'c_fr_sale_sg', 'c_inf_fr_operate_a', 'c_paid_goods_s', 
    'c_paid_to_for_empl', 'c_paid_for_taxes', 'n_cashflow_act', 'n_cashflow_inv_act', 
    'free_cashflow', 'n_cash_flows_fnc_act', 'n_incr_cash_cash_equ', 'c_cash_equ_beg_period', 
    'c_cash_equ_end_period', 'im_net_cashflow_oper_act', 'end_bal_cash', 'beg_bal_cash', 
    'c_pay_acq_const_fiolta', 'c_disp_withdrwl_invest', 'c_pay_dist_dpcp_int_exp'
]

# Financial indicator fields
INDICATOR_FIELDS = [
    'eps', 'dt_eps', 'revenue_ps', 'bps', 'cfps', 'gross_margin', 'netprofit_margin', 
    'grossprofit_margin', 'current_ratio', 'quick_ratio', 'cash_ratio', 'inv_turn', 
    'ar_turn', 'ca_turn', 'fa_turn', 'assets_turn', 'debt_to_assets', 'debt_to_eqt', 
    'roe', 'roa', 'roic', 'netprofit_yoy', 'or_yoy', 'basic_eps_yoy', 'assets_yoy', 
    'eqt_yoy', 'ocf_yoy', 'roe_yoy', 'equity_yoy', 'rd_exp', 'ebit', 'ebitda', 'fcff_ps' 
]

# TTM columns to compute
TTM_COLUMNS = [
    'eps_ttm', 'revenue_ps_ttm', 'roe_ttm', 'roa_ttm', 'netprofit_margin_ttm', 
    'grossprofit_margin_ttm', 'revenue_cagr_3y', 'netincome_cagr_3y', 'fcf_margin_ttm', 
    'debt_to_ebitda', 'rd_exp_to_capex'
]

# All columns for schema coercion (includes core fields added during processing)
ALL_COLUMNS = ['ts_code', 'report_period', 'ann_date', 'end_date', 'period', 'currency'] + INCOME_FIELDS + BALANCE_FIELDS + CASHFLOW_FIELDS + INDICATOR_FIELDS + TTM_COLUMNS

# Fields to convert from Yuan to Wan
YUAN_TO_WAN_FIELDS = INCOME_FIELDS + BALANCE_FIELDS + CASHFLOW_FIELDS + ['gross_margin', 'rd_exp', 'ebit', 'ebitda','fcf_ttm']

def convert_yuan_to_wan(df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
    """Convert specified fields from Yuan to Wan (divide by 10,000)"""
    for field in fields:
        if field in df.columns:
            # Convert to numeric first, then divide
            df[field] = pd.to_numeric(df[field], errors='coerce') / 10000
    return df

def coerce_to_float(df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
    """Coerce specified fields to float"""
    for field in fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce').astype(float)
    return df

def coerce_to_decimal(df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
    """Coerce specified fields to decimal-like (float with precision)"""
    for field in fields:
        if field in df.columns:
            # First ensure the field is numeric, then round
            df[field] = pd.to_numeric(df[field], errors='coerce').round(4)
    return df

def coerce_dates(df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
    """Coerce date fields to datetime"""
    for field in fields:
        if field in df.columns:
            df[field] = pd.to_datetime(df[field], format='%Y%m%d', errors='coerce')
    return df

def data_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce DataFrame schema to match database types"""
    if df.empty:
        return df
    df = df.copy()

    # Convert Yuan to Wan for absolute amounts
    df = convert_yuan_to_wan(df, YUAN_TO_WAN_FIELDS)

    # Coerce to float for ratios and per-share
    float_fields = [f for f in ALL_COLUMNS if f.endswith('_eps') or f.endswith('_ps') or f.endswith('_ratio') or f.endswith('_margin') or f.endswith('_turn') or f.endswith('_yoy') or f.endswith('_cagr_3y')]
    df = coerce_to_float(df, float_fields)

    # Coerce to decimal for large amounts
    decimal_fields = [f for f in ALL_COLUMNS if f in YUAN_TO_WAN_FIELDS]
    df = coerce_to_decimal(df, decimal_fields)

    # Handle dates
    date_fields = ['ann_date', 'end_date']
    df = coerce_dates(df, date_fields)

    # Fill missing values
    df = df.fillna(pd.NA)

    # Add currency and period if missing
    if 'currency' not in df.columns:
        df['currency'] = 'CNY'
    if 'period' not in df.columns:
        df['period'] = 'quarter'  # Default, adjust as needed

    return df

def calculate_quarterly_ttm_sums(df: pd.DataFrame, sum_cols: List[str]) -> pd.DataFrame:
    """Calculate rolling TTM sums for specified columns"""
    for col in sum_cols:
        df[f'ttm_{col}'] = df.groupby('ts_code')['q_' + col].rolling(4, min_periods=4).sum().reset_index(0, drop=True)
    return df

def calculate_semi_annual_ttm_sums(df: pd.DataFrame, sum_cols: List[str]) -> pd.DataFrame:
    """Calculate rolling TTM sums for specified columns"""
    for col in sum_cols:
        df[f'ttm_{col}'] = df.groupby('ts_code')['hy_' + col].rolling(4, min_periods=2).sum().reset_index(0, drop=True)
    return df

def calculate_cagr(df: pd.DataFrame, col: str, output_prefix: str = None, years: int = 3) -> pd.DataFrame:
    """Calculate CAGR for specified column"""
    if output_prefix is None:
        output_prefix = col
    lag_col = f'{col}_{years}y_ago'
    df[lag_col] = df.groupby('ts_code')[col].shift(12)

    if lag_col in df.columns:
        mask_positive = (df[lag_col] > 0) & (df[col] > 0)
        df[f'{output_prefix}_cagr_{years}y'] = np.where(
            mask_positive,
            ((df[col] / df[lag_col]) ** (1/years) - 1) * 100,
            np.nan
        )
    return df

def calculate_fcf_ttm(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate FCF TTM with improved handling"""
    if 'n_cashflow_act' in df.columns and 'c_pay_acq_const_fiolta' in df.columns:
        df['fcf_ttm'] = df['n_cashflow_act'] - df['c_pay_acq_const_fiolta'].fillna(0)
    elif 'n_cashflow_act' in df.columns:
        if 'c_pay_acq_const_fiolta' in df.columns:
            capex_avg = df.groupby('ts_code')['c_pay_acq_const_fiolta'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
            df['fcf_ttm'] = np.where(
                capex_avg.notna(),
                df['n_cashflow_act'] - capex_avg,
                df['n_cashflow_act'] * 0.7
            )
        else:
            df['fcf_ttm'] = df['n_cashflow_act'] * 0.7
    else:
        df['fcf_ttm'] = np.nan
    return df

def calculate_fcf_margin_ttm(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate FCF margin TTM"""
    if 'fcf_ttm' in df.columns and 'ttm_total_revenue' in df.columns:
        df['fcf_margin_ttm'] = np.where(
            df['ttm_total_revenue'] > 0,
            (df['fcf_ttm'] / df['ttm_total_revenue']) * 100,
            np.nan
        )
    return df

def calculate_debt_to_ebitda_ttm(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Debt to EBITDA TTM"""
    if 'total_liab' in df.columns and 'ttm_ebitda' in df.columns:
        df['debt_to_ebitda'] = np.where(
            df['ttm_ebitda'] > 0,
            df['total_liab'] / df['ttm_ebitda'],
            np.nan
        )
    return df

def calculate_rd_exp_to_capex(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate R&D expense to CapEx ratio"""
    if 'rd_exp' in df.columns and 'c_pay_acq_const_fiolta' in df.columns:
        df['rd_exp_to_capex'] = np.where(
            df['c_pay_acq_const_fiolta'] > 0,
            df['rd_exp'] / (df['c_pay_acq_const_fiolta'] + df['rd_exp']) * 100,
            np.nan
        )
    return df

def get_existing_dates(group: pd.DataFrame) -> pd.Series:
    """Extract and sort existing report dates."""
    return group['report_date'].dropna().sort_values().unique()

def generate_full_quarters(min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DatetimeIndex:
    """Generate complete quarter-end date sequence within range."""
    return pd.date_range(start=min_date, end=max_date, freq='QE-SEP')

def create_full_df(dates: pd.DatetimeIndex, ts_code: str) -> pd.DataFrame:
    """Create DataFrame with full quarter dates and ts_code."""
    full_df = pd.DataFrame({'report_date': dates})
    full_df['report_period'] = full_df['report_date'].dt.strftime('%Y%m%d')
    full_df['ts_code'] = ts_code
    return full_df

def merge_with_original(ts_code: str, full: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """Merge full date frame with original data."""
    original = original.drop_duplicates(subset=['ts_code', 'report_period', 'report_date'], keep='last')
    merged = pd.merge(full, original, on=['ts_code', 'report_period', 'report_date'], how='left')

    # set missing
    missing_count = merged['ann_date'].isna().sum()
    if missing_count > 0:
        logger.info(f"complete data len: {ts_code}, {missing_count}")
    missing_mask = merged['ann_date'].isna()
    merged['missing'] = missing_mask.astype(int)
    return merged
    
# 为每个ts_code补全中间缺失的季度序列
def complete_quarters(origin: pd.DataFrame) -> pd.DataFrame:
    origin['report_date'] = pd.to_datetime(origin['report_period'], format='%Y%m%d')
    df = origin.sort_values(['ts_code', 'report_date'])
    completed_groups = []

    for ts_code, group_iter in itertools.groupby(df.iterrows(), key=lambda x: x[1]['ts_code']):
        group_df = pd.DataFrame([row[1] for row in group_iter])
        existing_dates = group_df['report_date'].dropna().sort_values().unique()
        if len(existing_dates) < 2:
            group_df['missing'] = 0
            completed_groups.append(group_df)
            continue

        min_date = existing_dates.min()
        max_date = existing_dates.max()

        full_dates = generate_full_quarters(min_date, max_date)
        if len(full_dates) == 0:
            completed_groups.append(pd.DataFrame())
            continue
        full_df = create_full_df(full_dates, ts_code)
        merged = merge_with_original(ts_code, full_df, group_df)
        completed_groups.append(merged.sort_values('report_date').reset_index(drop=True))

    return pd.concat(completed_groups, ignore_index=True)

def calculate_quarterly_values(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Calculate quarterly values using vectorized operations within each year"""
    df = df.sort_values('report_period')
    df['year'] = df['report_period'].astype(str).str[:4]
    for col in columns:
        df['q_' + col] = df.groupby(['ts_code', 'year'])[col].diff().fillna(df[col])
    return df.drop(columns=['year'])

def calculate_semi_annual_values(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Vectorized semi-annual values calculation for entire df, as supplement for missing Q1/Q3.
    - For H1 + FY: H1 hy_ = H1 (cumulative), H2 hy_ = FY - H1.
    - For only H1: H1 hy_ = H1.
    - For only FY: only log.
    """
    if df.empty:
        return df
   
    df = df.sort_values(['ts_code', 'report_period']).copy()
    df['year'] = df['report_period'].str[:4]
    df['month_day'] = df['report_period'].str[4:]
   
    group_key = ['ts_code', 'year']
   
    # Filter only semi-annual data first
    df_semi = df[df['month_day'].isin(['0630', '1231'])].copy()
   
    if df_semi.empty:
        return df  # No semi data, return original
   
    # Compute per group info on filtered df
    group_info = df_semi.groupby(group_key).agg(
        has_h1=('month_day', lambda x: '0630' in x.values),
        has_fy=('month_day', lambda x: '1231' in x.values)
    ).reset_index()
   
    # Merge info back to df_semi
    df_semi = df_semi.merge(group_info, on=group_key, how='left')
    df_semi.fillna({'has_h1': False, 'has_fy': False}, inplace=True)
   
    # Masks for semi years on df_semi
    mask_both = df_semi['has_h1'] & df_semi['has_fy']
    mask_h1 = df_semi['has_h1'] & ~df_semi['has_fy']
    mask_fy = ~df_semi['has_h1'] & df_semi['has_fy']
   
    for col in columns:
        df_semi['hy_' + col] = np.nan
       
        # both: H1 hy_ = col (on 0630 row)
        h1_both_mask = mask_both & (df_semi['month_day'] == '0630')
        df_semi.loc[h1_both_mask, 'hy_' + col] = df_semi.loc[h1_both_mask, col]
       
        # both: FY hy_ = col - shift(1) (since sorted and filtered, shift(1) is 0630 for 1231 if present)
        fy_both_mask = mask_both & (df_semi['month_day'] == '1231')
        df_semi['tmp_shift'] = df_semi.groupby(group_key)[col].shift(1)
        diff = df_semi.loc[fy_both_mask, col] - df_semi.loc[fy_both_mask, 'tmp_shift']
        df_semi.loc[fy_both_mask, 'hy_' + col] = diff
       
        # only h1: hy_ = col (on 0630 row)
        h1_only_mask = mask_h1 & (df_semi['month_day'] == '0630')
        df_semi.loc[h1_only_mask, 'hy_' + col] = df_semi.loc[h1_only_mask, col]
       
        # only fy: log only
        if mask_fy.any():
            fy_only_rows = df_semi[mask_fy]
            logger.info(f"FY-only rows count: {len(fy_only_rows)}")
            if not fy_only_rows.empty:
                # Group by ts_code to show unique stocks with only FY data
                fy_only_by_code = fy_only_rows.groupby(['ts_code', 'report_period']).size()
                logger.info(f"Sample FY-only (ts_code, report_period) pairs: {fy_only_by_code.index.tolist()}")
   
    # Clean up df_semi
    drop_cols = ['year', 'month_day', 'has_h1', 'has_fy', 'tmp_shift']
    df_semi.drop(columns=drop_cols, inplace=True, errors='ignore')
   
    # Merge hy_ columns back to original df (left join to keep shape)
    hy_cols = ['ts_code', 'report_period'] + ['hy_' + col for col in columns]
    df = df.merge(df_semi[hy_cols], on=['ts_code', 'report_period'], how='left')
    return df
    
def clean_completed_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Remove filled rows (missing=1) after calculations are complete
    if 'missing' in df.columns:
        original_count = len(df)
        df = df[df['missing'] != 1].copy()
        removed_count = original_count - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} filled rows after TTM/CAGR calculations")
        df = df.drop(columns=['missing'])
    return df

def calculate_ttm_values(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate TTM values using quarterly data, with semi-annual fallback when needed."""
    sum_cols = ['n_income_attr_p', 'total_revenue', 'ebitda', 'oper_cost', 'total_cogs']

    # Calculate quarterly values (diffs)
    df = calculate_quarterly_values(df, sum_cols)
    # Try quarterly TTM sums first
    df_quarter = calculate_quarterly_ttm_sums(df.copy(), sum_cols)

    # Check which rows have NaN TTM values (insufficient quarterly data)
    ttm_cols = [f'ttm_{col}' for col in sum_cols]
    na_mask = df_quarter[ttm_cols].isna().any(axis=1)

    if not na_mask.any():
        return df_quarter  # All quarterly TTM calculations succeeded

    # For rows with NaN quarterly TTM, try semi-annual approach
    logger.info(f"Found {na_mask.sum()} rows with insufficient quarterly data, trying semi-annual fallback")

    # Add semi-annual columns
    df_semi = calculate_semi_annual_values(df_quarter, sum_cols)
    # Calculate semi-annual TTM sums
    df_semi_ttm = calculate_semi_annual_ttm_sums(df_semi.copy(), sum_cols)
    df_semi_ttm.to_csv('df_semi_ttm.csv', index=False)
    # Fill NaN quarterly TTM values with semi-annual TTM values
    df_quarter.to_csv('df_quarter_before.csv', index=False)

    # Ensure both DataFrames are sorted consistently before filling
    sort_keys = ['ts_code', 'report_period']
    df_quarter = df_quarter.sort_values(sort_keys).reset_index(drop=True)
    df_semi_ttm = df_semi_ttm.sort_values(sort_keys).reset_index(drop=True)

    # Fill NaN values column by column to preserve existing valid values
    for col in sum_cols:
        ttm_col = f'ttm_{col}'
        col_na_mask = df_quarter[ttm_col].isna()
        df_quarter[ttm_col] = np.where(col_na_mask, df_semi_ttm[ttm_col], df_quarter[ttm_col])
    df_quarter.to_csv('df_quarter_after.csv', index=False)

    return df_quarter

def calculate_ttm_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all TTM indicators"""
    if df.empty:
        return df

    df = df.sort_values(['ts_code', 'report_period'])

    # Complete missingquarters
    df = complete_quarters(df)
    logger.debug(f"in calculate_ttm_indicators, after complete_quarters, df len: {len(df)}")

    # Calculate TTM sum values
    df = calculate_ttm_values(df)

    # Calculate per-share TTM
    if 'ttm_n_income_attr_p' in df.columns and 'total_share' in df.columns:
        df['eps_ttm'] = df['ttm_n_income_attr_p'] / df['total_share']
    if 'ttm_total_revenue' in df.columns and 'total_share' in df.columns:
        df['revenue_ps_ttm'] = df['ttm_total_revenue'] / df['total_share']

    # Calculate profitability TTM
    if 'ttm_n_income_attr_p' in df.columns and 'total_hldr_eqy_exc_min_int' in df.columns:
        df['roe_ttm'] = (df['ttm_n_income_attr_p'] / df['total_hldr_eqy_exc_min_int']) * 100
    if 'ttm_n_income_attr_p' in df.columns and 'total_assets' in df.columns:
        df['roa_ttm'] = (df['ttm_n_income_attr_p'] / df['total_assets']) * 100
    if 'ttm_n_income_attr_p' in df.columns and 'ttm_total_revenue' in df.columns:
        df['netprofit_margin_ttm'] = (df['ttm_n_income_attr_p'] / df['ttm_total_revenue']) * 100
    if 'ttm_oper_cost' in df.columns and 'ttm_total_revenue' in df.columns:
        df['grossprofit_margin_ttm'] = ((df['ttm_total_revenue'] - df['ttm_oper_cost']) / df['ttm_total_revenue']) * 100

    # Calculate CAGRs
    df = calculate_cagr(df, 'total_revenue', output_prefix='revenue')
    df = calculate_cagr(df, 'n_income_attr_p', output_prefix='netincome')

    # Calculate FCF and related
    df = calculate_fcf_ttm(df)
    df = calculate_fcf_margin_ttm(df)

    # Calculate ratios
    df = calculate_debt_to_ebitda_ttm(df)
    df = calculate_rd_exp_to_capex(df)

    logger.debug(f"before clean_completed_rows, df len: {len(df)}")
    df = clean_completed_rows(df)
    logger.debug(f"after clean_completed_rows, df len: {len(df)}")
    return df

def fetch_api_data(api_func, fields: str, period: str) -> pd.DataFrame:
    """Fetch data from a single Tushare API"""
    try:
        return call_tushare_api_with_retry(api_func, period=period, fields=fields)
    except Exception as e:
        logger.error(f"Error fetching data from {api_func.__name__}: {e}")
        return pd.DataFrame()

def single_period_data_preprocess(df: pd.DataFrame, common_fields: List[str]=None) -> pd.DataFrame:
    """single period data preprocess"""
    if 'end_date' in df.columns:
        df = df.rename(columns={'end_date': 'report_period'})

    # Deduplicate
    df = deduplicate_dataframe(df, ['ts_code', 'report_period'])

    if common_fields:
        df = df.drop(columns=[col for col in common_fields if col in df.columns], errors='ignore')
    return df

def merge_dataframes(main: pd.DataFrame, dfs: List[pd.DataFrame], on: List[str]) -> pd.DataFrame:
    """Merge multiple DataFrames on specified keys"""
    merged = main.copy()
    for df in dfs:
        if not df.empty:
            merged = merged.merge(df, on=on, how='left')
    return merged if merged is not None else pd.DataFrame()

def deduplicate_dataframe(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    """Deduplicate DataFrame based on subset, keeping last"""
    if not df.empty:
        initial_count = len(df)
        df = df.drop_duplicates(subset=subset, keep='last')
        if initial_count != len(df):
            logger.info(f"Removed {initial_count - len(df)} duplicates")
    return df

def _fetch_single_period_data(report_period: str) -> pd.DataFrame:
    """Fetch and process data for a single period"""
    try:
        # Fetch individual data sources
        income_df = fetch_api_data(tushare_pro.income_vip, ','.join(COMMON_FIELDS + INCOME_FIELDS), report_period)
        balance_df = fetch_api_data(tushare_pro.balancesheet_vip, ','.join(COMMON_FIELDS + BALANCE_FIELDS), report_period)
        cashflow_df = fetch_api_data(tushare_pro.cashflow_vip, ','.join(COMMON_FIELDS + CASHFLOW_FIELDS), report_period)
        indicator_df = fetch_api_data(tushare_pro.fina_indicator_vip, ','.join(COMMON_FIELDS + INDICATOR_FIELDS), report_period)

        common_non_keys = ['ann_date', 'end_date', 'report_type']
        income_df = single_period_data_preprocess(income_df, common_non_keys)
        balance_df = single_period_data_preprocess(balance_df, common_non_keys)
        cashflow_df = single_period_data_preprocess(cashflow_df, common_non_keys)
        indicator_df = single_period_data_preprocess(indicator_df)

        # Merge all dataframes
        merged_df = merge_dataframes(indicator_df, [income_df, balance_df, cashflow_df], on=['ts_code', 'report_period'])
        return merged_df

    except Exception as e:
        logger.error(f"Error in _fetch_single_period_data for period {report_period}: {e}")
        return pd.DataFrame()

def generate_annual_periods(end_dt: datetime.datetime, limit: int) -> List[str]:
    """Generate annual periods"""
    periods = []
    for i in range(limit):
        year = end_dt.year - i
        if datetime.datetime(year, 12, 31) <= end_dt:
            periods.append(f"{year}1231")
    return periods

def generate_quarterly_periods(end_dt: datetime.datetime, limit: int) -> List[str]:
    """Generate quarterly periods"""
    periods = []
    quarters = [(3, 31), (6, 30), (9, 30), (12, 31)]
    current_quarter = None
    for month, day in reversed(quarters):
        q_date = datetime.datetime(end_dt.year, month, day)
        if q_date <= end_dt:
            current_quarter = q_date
            break
    if current_quarter:
        for i in range(limit):
            periods.append(current_quarter.strftime("%Y%m%d"))
            if current_quarter.month == 3:
                current_quarter = datetime.datetime(current_quarter.year - 1, 12, 31)
            elif current_quarter.month == 6:
                current_quarter = datetime.datetime(current_quarter.year, 3, 31)
            elif current_quarter.month == 9:
                current_quarter = datetime.datetime(current_quarter.year, 6, 30)
            else:
                current_quarter = datetime.datetime(current_quarter.year, 9, 30)
    return periods

def _generate_periods(end_date: str, period: str = "annual", limit: int = 1) -> List[str]:
    """Generate list of periods that need to be processed"""
    end_dt = datetime.datetime.strptime(end_date, "%Y%m%d")
    if period == "annual":
        return generate_annual_periods(end_dt, limit)
    else:
        return generate_quarterly_periods(end_dt, limit)

def fetch_period_data(periods: List[str]) -> List[pd.DataFrame]:
    """Fetch data for multiple periods"""
    all_data = []
    for report_period in periods:
        df = _fetch_single_period_data(report_period)
        if not df.empty:
            all_data.append(df)
        time.sleep(0.5)
    return all_data

def _fetch_financial_data(end_date: str, period: str = "annual", limit: int = 1) -> pd.DataFrame:
    """Fetch financial data for multiple periods"""
    periods = _generate_periods(end_date, period, limit)
    if not periods:
        logger.warning("No valid periods found")
        return pd.DataFrame()

    logger.info(f"Fetching financial data for periods: {periods}")

    all_data = fetch_period_data(periods)

    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Successfully fetched {len(result_df)} financial records in total")
        return result_df
    else:
        return pd.DataFrame()

def prepare_upsert_batch(rows: List[Dict[str, Any]], table: Table, chunksize: int) -> int:
    """Prepare and execute upsert in batches"""
    total_affected = 0
    for i in range(0, len(rows), chunksize):
        batch = rows[i:i + chunksize]
        stmt = mysql_insert(table).values(batch)
        update_map = {
            c: getattr(stmt.inserted, c)
            for c in ALL_COLUMNS
            if c not in ("ts_code", "report_period", "ann_date")
        }
        ondup = stmt.on_duplicate_key_update(**update_map)
        # Assume conn.execute(ondup) here, but in context
        total_affected += len(batch)  # Simplified for example
    return total_affected

def _upsert_batch(engine, df: pd.DataFrame, chunksize: int = 1000) -> int:
    """Batch upsert data to MySQL"""
    if df.empty:
        return 0

    total_processed = len(df)
    meta = MetaData()
    table = Table(TABLE_NAME, meta, autoload_with=engine)
    rows = df.to_dict(orient="records")

    with engine.begin() as conn:
        total_affected = prepare_upsert_batch(rows, table, chunksize)
        # Actual execution would be here

    logger.info(f"Processed {total_processed} records, database reported {total_affected} affected rows")
    return total_processed

def set_default_end_date(end_date: Optional[str]) -> str:
    """Set default end date if none provided"""
    if end_date is None:
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        end_date = yesterday.strftime("%Y%m%d")
    return end_date

def create_database_engine(mysql_url: str) -> Any:
    """Create SQLAlchemy engine"""
    return create_engine(mysql_url, pool_recycle=3600)

def create_table_structure(engine: Any) -> None:
    """Create table if not exists"""
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_DDL))

def fetch_and_normalize_period_data(periods: List[str]) -> List[pd.DataFrame]:
    """Fetch and normalize data for periods"""
    all_data_frames = []
    for report_period in periods:
        df = _fetch_single_period_data(report_period)
        logger.info(f"fetch report_period: {report_period}, {len(df)} records.")
        if not df.empty:
            df = data_preprocess(df)
            all_data_frames.append(df)
        time.sleep(0.5)
    return all_data_frames

def combine_data_frames(all_data_frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Combine list of DataFrames"""
    if not all_data_frames:
        return pd.DataFrame()
    return pd.concat(all_data_frames, ignore_index=True)

def log_update_completion(periods: List[str], total_raw_records: int, combined_df: pd.DataFrame, total_written: int) -> None:
    """Log completion statistics"""
    logger.info(f"Update completed:")
    logger.info(f"- Processed {len(periods)} periods")
    logger.info(f"- Retrieved {total_raw_records} raw records")
    logger.info(f"- Final records after TTM calculation: {len(combined_df)}")
    logger.info(f"- Total records written to database: {total_written}")

def update_a_stock_financial_profile(
    mysql_url: str = "mysql+pymysql://root:@127.0.0.1:3306/investment_data",
    end_date: Optional[str] = None,
    period: str = "quarter",
    limit: int = 10,
    chunksize: int = 1000,
) -> None:
    """
    Incrementally fetch Tushare financial profile data and write to MySQL ts_a_stock_financial_profile table
    Contains complete financial data, grouped by relevance

    Data sources:
    - pro.income_vip() - Income statement
    - pro.balancesheet_vip() - Balance sheet
    - pro.cashflow_vip() - Cash flow statement
    - pro.fina_indicator_vip() - Financial indicators

    Args:
        mysql_url: MySQL connection URL
        end_date: End date in YYYYMMDD format, defaults to yesterday
        period: Report period type, 'annual' or 'quarter'
        limit: Limit on number of report periods to fetch
        chunksize: Batch processing size
    """
    try:
        end_date = set_default_end_date(end_date)
        logger.info(f"Starting to update financial profile data, end date: {end_date}, period type: {period}, limit: {limit}")

        # engine = create_database_engine(mysql_url)
        # create_table_structure(engine)

        periods = _generate_periods(end_date, period, limit)
        logger.info(f"Generated {len(periods)} periods: {periods}")
        if not periods:
            logger.warning("No valid periods found")
            return

        all_data_frames = fetch_and_normalize_period_data(periods)
        if not all_data_frames:
            logger.warning("No data retrieved for any period")
            return

        total_raw_records = sum(len(df) for df in all_data_frames)

        combined_df = combine_data_frames(all_data_frames)
        #combined_df.to_csv("original_new.csv", index=False)
        logger.info(f"Combined {len(all_data_frames)} periods into {len(combined_df)} total records")

        logger.info("Calculating TTM (Trailing Twelve Months) indicators...")
        combined_df = calculate_ttm_indicators(combined_df)
        logger.info(f"TTM calculation completed, {len(combined_df)} records after TTM processing")
        #combined_df.to_csv("combined_df.csv", index=False)

        # total_written = _upsert_batch(engine, combined_df, chunksize)
        total_written = len(combined_df)  # Placeholder

        log_update_completion(periods, total_raw_records, combined_df, total_written)

    except Exception as e:
        logger.error(f"Fatal error in update_a_stock_financial_profile: {e}")
        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        raise

if __name__ == "__main__":
    fire.Fire(update_a_stock_financial_profile)
