/* ============================================================================
   Migration Script for final_a_stock_comb_info table

   This script migrates data from the old table structure to the new structure.
   The new structure includes updated financial metrics and removes outdated fields.

   IMPORTANT: This script should be run after backing up the existing table.

   ============================================================================ */

/* Step 1: Create new table structure */
SELECT "Creating new table structure..." as info;
DROP TABLE IF EXISTS final_a_stock_comb_info_new;

CREATE TABLE final_a_stock_comb_info_new (
  tradedate DATE NOT NULL,
  symbol VARCHAR(16) NOT NULL,
  industry INT,
  high FLOAT,
  low FLOAT,
  open FLOAT,
  close FLOAT,
  volume BIGINT UNSIGNED,
  adjclose FLOAT,
  amount BIGINT UNSIGNED,
  turnover_rate FLOAT,
  volume_ratio FLOAT,
  pe FLOAT,
  pb FLOAT,
  ps FLOAT,
  dv_ratio FLOAT,
  circ_mv BIGINT UNSIGNED,
  main_inflow_ratio FLOAT,
  small_inflow_ratio FLOAT,
  net_inflow_ratio FLOAT,
  cost_5pct FLOAT,
  cost_15pct FLOAT,
  cost_50pct FLOAT,
  cost_85pct FLOAT,
  cost_95pct FLOAT,
  weight_avg FLOAT,
  winner_rate FLOAT,
  f_pos_ratio FLOAT,
  f_neg_ratio FLOAT,
  f_target_price FLOAT,
  f_eps FLOAT,
  f_pe FLOAT,
  f_dv_ratio FLOAT,
  f_roe FLOAT,

  /* Liquidity */
  current_ratio FLOAT,
  quick_ratio FLOAT,
  cash_ratio FLOAT,
  /* Efficiency & Turnover */
  ca_turn FLOAT,
  inv_turn FLOAT,
  ar_turn FLOAT,
  fa_turn FLOAT,
  assets_turn FLOAT,
  /* Profitability */
  roic FLOAT,
  roe_ttm FLOAT,
  roa_ttm FLOAT,
  grossprofit_margin_ttm FLOAT,
  netprofit_margin_ttm FLOAT,
  fcf_margin_ttm FLOAT,
  /* Growth */
  or_yoy FLOAT,
  netprofit_yoy FLOAT,
  basic_eps_yoy FLOAT,
  equity_yoy FLOAT,
  assets_yoy FLOAT,
  ocf_yoy FLOAT,
  roe_yoy FLOAT,
  revenue_cagr_3y FLOAT,
  netincome_cagr_3y FLOAT,
  /* Leverage & Risk*/
  debt_to_assets FLOAT,
  debt_to_eqt FLOAT,
  debt_to_ebitda FLOAT,
  /* Valuation & Cash Flow */
  bps FLOAT,
  eps_ttm FLOAT,
  revenue_ps_ttm FLOAT,
  cfps FLOAT,
  fcff_ps FLOAT,
  /* Other */
  rd_exp_to_capex FLOAT,
  goodwill FLOAT,
  PRIMARY KEY (tradedate, symbol),
  INDEX idx_tradedate_desc (tradedate DESC),
  INDEX idx_comb_symbol_tradedate (symbol, tradedate)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;

/* Step 2: Migrate data from old table to new table */
SELECT "Migrating data from old table to new table..." as info;

INSERT INTO final_a_stock_comb_info_new (
  tradedate, symbol, high, low, open, close, volume, adjclose, amount,
  turnover_rate, volume_ratio, pe, pb, ps, dv_ratio, circ_mv,
  main_inflow_ratio, small_inflow_ratio, net_inflow_ratio,
  cost_5pct, cost_15pct, cost_50pct, cost_85pct, cost_95pct,
  weight_avg, winner_rate, f_pos_ratio, f_neg_ratio, f_target_price,
  f_eps, f_pe, f_dv_ratio, f_roe
)
SELECT
  tradedate, symbol, high, low, open, close, volume, adjclose, amount,
  turnover_rate, volume_ratio, pe, pb, ps, dv_ratio, circ_mv,
  main_inflow_ratio, small_inflow_ratio, net_inflow_ratio,
  cost_5pct, cost_15pct, cost_50pct, cost_85pct, cost_95pct,
  weight_avg, winner_rate, f_pos_ratio, f_neg_ratio, f_target_price,
  f_eps, f_pe, f_dv_ratio, f_roe
FROM final_a_stock_comb_info;

/* Step 3: Verify migration */
SELECT "Migration verification:" as info;
SELECT
    (SELECT COUNT(*) FROM final_a_stock_comb_info) as old_table_count,
    (SELECT COUNT(*) FROM final_a_stock_comb_info_new) as new_table_count;

/* Show sample of migrated data */
SELECT "Sample of migrated data:" as info;
SELECT
    tradedate,
    symbol,
    close,
    turnover_rate,
    pe,
    circ_mv,
    main_inflow_ratio,
    cost_50pct,
    winner_rate,
    f_pos_ratio,
    f_target_price
FROM final_a_stock_comb_info_new
WHERE turnover_rate IS NOT NULL OR pe IS NOT NULL OR circ_mv IS NOT NULL OR main_inflow_ratio IS NOT NULL OR cost_50pct IS NOT NULL OR winner_rate IS NOT NULL OR f_pos_ratio IS NOT NULL OR f_target_price IS NOT NULL
ORDER BY tradedate DESC, symbol
LIMIT 5;

/* Step 4: Switch tables */
SELECT "Switching tables..." as info;
RENAME TABLE final_a_stock_comb_info TO final_a_stock_comb_info_old;
RENAME TABLE final_a_stock_comb_info_new TO final_a_stock_comb_info;

SELECT "Table switch completed. Old table renamed to final_a_stock_comb_info_old" as info;

/* Alternative: Drop old table after verification (uncomment when ready) */
/*
SELECT "Dropping old table after verification..." as info;
DROP TABLE final_a_stock_comb_info;
RENAME TABLE final_a_stock_comb_info_new TO final_a_stock_comb_info;

SELECT "Migration completed successfully!" as info;
*/

SELECT "Migration script completed. Please verify data before switching tables." as info;
