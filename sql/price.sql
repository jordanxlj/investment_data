/* ============================================================================
   Price Update Script for final_a_stock_comb_info table

   ============================================================================ */
/* Create final table for combined info if it does not exist
   - percentages/ratios stored as FLOAT (already divided by 100 in SELECT)
   - shares/market cap stored in base units (×10000), as BIGINT UNSIGNED */
CREATE TABLE IF NOT EXISTS final_a_stock_comb_info (
  tradedate DATE NOT NULL,
  symbol VARCHAR(16) NOT NULL,
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
  gross_margin_ttm FLOAT,
  operating_margin_ttm FLOAT,
  net_margin_ttm FLOAT,
  roe_ttm FLOAT,
  roa_ttm FLOAT,
  roic FLOAT,
  debt_to_equity_ttm FLOAT,
  debt_to_assets_ttm FLOAT,
  current_ratio_ttm FLOAT,
  quick_ratio FLOAT,
  cash_ratio FLOAT,
  revenue_cagr_3y FLOAT,
  net_income_cagr_3y FLOAT,
  rd_exp_to_capex FLOAT,
  eps_cagr_3y FLOAT,
  ebitda_cagr_3y FLOAT,
  operating_income_cagr_3y FLOAT,
  gross_profit_cagr_3y FLOAT,
  free_cash_flow_cagr_3y FLOAT,
  dividend_cagr_3y FLOAT,
  total_assets_cagr_3y FLOAT,
  PRIMARY KEY (tradedate, symbol),
  INDEX idx_tradedate_desc (tradedate DESC),
  INDEX idx_comb_symbol_tradedate (symbol, tradedate)
) ENGINE=InnoDB ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;


/* Set shared variables to avoid repeated subqueries */
SET @max_tradedate = (SELECT COALESCE(MAX(tradedate), '2008-01-01') FROM final_a_stock_comb_info);
SET @start_date = '2025-09-01';  /* Start date for data processing - matches consensus data */
SET @debug = 0;  /* Set to 1 to enable debug output */

SELECT CONCAT('Optimization: Using max_tradedate = ', @max_tradedate, ', start_date = ', @start_date, ', debug = ', @debug) AS optimization_info;


/* Add new stock to ts_link_table */
SELECT "Add new stock to ts_link_table" as info;
INSERT IGNORE INTO ts_link_table (w_symbol, link_symbol, link_date)
select concat(substr(symbol, 8, 2), substr(symbol, 1, 6)) as w_symbol, symbol as link_symbol, max(tradedate) as link_date 
from ts_a_stock_eod_price 
where tradedate = (select max(tradedate) from ts_a_stock_eod_price) group by symbol;

/* Fill in new stock price - Initial population for empty table */
/* This query handles the initial population when final_a_stock_comb_info is empty */
SELECT "Fill in new stock price - Initial population for empty table" as info;
INSERT IGNORE INTO final_a_stock_comb_info (tradedate, symbol, high, low, open, close, volume, adjclose, amount)
SELECT ts_a_stock_eod_price.tradedate,
       ts_link_table.w_symbol as symbol,
       ts_a_stock_eod_price.high,
       ts_a_stock_eod_price.low,
       ts_a_stock_eod_price.open,
       ts_a_stock_eod_price.close,
       ts_a_stock_eod_price.volume,
       ROUND(ts_a_stock_eod_price.adjclose, 2),
       ts_a_stock_eod_price.amount
FROM ts_a_stock_eod_price
INNER JOIN ts_link_table ON ts_a_stock_eod_price.symbol = ts_link_table.link_symbol
LEFT JOIN final_a_stock_comb_info existing ON existing.symbol = ts_link_table.w_symbol
  AND existing.tradedate = ts_a_stock_eod_price.tradedate
WHERE existing.symbol IS NULL  -- Only insert records that don't exist
  AND ts_a_stock_eod_price.tradedate >= @start_date;

/* Set new stock adj ratio to 1 */
UPDATE ts_link_table  SET adj_ratio=1 WHERE adj_ratio is NULL;

/* Fill in index price from ts */
SELECT "Fill in index price from ts" as info;
INSERT IGNORE INTO final_a_stock_comb_info (tradedate, symbol, high, low, open, close, volume, adjclose, amount)
select ts_raw_table.tradedate,
			ts_link_table.w_symbol as symbol,
			ts_raw_table.high,
			ts_raw_table.low,
			ts_raw_table.open,
			ts_raw_table.close,
			ts_raw_table.volume,
			ROUND(ts_raw_table.adjclose / ts_link_table.adj_ratio, 2) as adjclose,
			ts_raw_table.amount
FROM (
SELECT * FROM ts_a_stock_eod_price
WHERE tradedate >= @start_date
  AND tradedate >
		(
			select max(tradedate) as tradedate
			FROM final_a_stock_comb_info
			where symbol = "SZ399300"
		)
) ts_raw_table
LEFT JOIN ts_link_table ON ts_raw_table.symbol = ts_link_table.link_symbol;

/* Fill in stock price from ts - only for incremental updates when table is not empty */
SELECT "Fill in stock price from ts - only for incremental updates when table is not empty" as info;
INSERT IGNORE INTO final_a_stock_comb_info (tradedate, symbol, high, low, open, close, volume, adjclose, amount)
select ts_raw_table.tradedate,
			ts_link_table.w_symbol as symbol,
			ts_raw_table.high,
			ts_raw_table.low,
			ts_raw_table.open,
			ts_raw_table.close,
			ts_raw_table.volume,
			ROUND(ts_raw_table.adjclose / ts_link_table.adj_ratio, 2) as adjclose,
			ts_raw_table.amount
FROM (
SELECT * FROM ts_a_stock_eod_price
WHERE tradedate >= @start_date
  AND tradedate > COALESCE((
	select max(tradedate) as tradedate
	FROM
		(select tradedate, count(tradedate) as symbol_count
		FROM final_a_stock_comb_info
		where tradedate >= @start_date
		group by tradedate) tradedate_record
	WHERE symbol_count > 1000
  ), '1900-01-01')  -- If table is empty, this will prevent any inserts
) ts_raw_table
LEFT JOIN ts_link_table ON ts_raw_table.symbol = ts_link_table.link_symbol
LEFT JOIN final_a_stock_comb_info existing ON existing.symbol = ts_link_table.w_symbol
  AND existing.tradedate = ts_raw_table.tradedate
WHERE existing.symbol IS NULL;  -- Double-check to prevent duplicates
