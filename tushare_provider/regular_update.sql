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
  roic_ttm FLOAT,
  revenue_growth_ttm FLOAT,
  revenue_cagr_3y FLOAT,
  net_income_cagr_3y FLOAT,
  debt_to_equity_ttm FLOAT,
  debt_to_assets_ttm FLOAT,
  current_ratio_ttm FLOAT,
  quick_ratio_ttm FLOAT,
  cash_ratio_ttm FLOAT,
  suspend BOOL,
 PRIMARY KEY (tradedate, symbol),
 INDEX idx_tradedate (tradedate),
 INDEX idx_symbol (symbol),
 INDEX idx_comb_symbol_tradedate (symbol, tradedate)
) ENGINE=InnoDB ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;


/* Add new stock to ts_link_table */
INSERT IGNORE INTO ts_link_table (w_symbol, link_symbol, link_date)
select concat(substr(symbol, 8, 2), substr(symbol, 1, 6)) as w_symbol, symbol as link_symbol, max(tradedate) as link_date 
from ts_a_stock_eod_price 
where tradedate = (select max(tradedate) from ts_a_stock_eod_price) group by symbol;

/* Fill in new stock price */
/* Fill in stock where w stock does not exists */
INSERT IGNORE INTO final_a_stock_comb_info (tradedate, symbol, high, low, open, close, volume, adjclose, amount)
select ts_a_stock_eod_price.tradedate, 
			missing_table.w_symbol as symbol,
			ts_a_stock_eod_price.high,
			ts_a_stock_eod_price.low,
			ts_a_stock_eod_price.open,
			ts_a_stock_eod_price.close,
			ts_a_stock_eod_price.volume,
			ROUND(ts_a_stock_eod_price.adjclose, 2),
			ts_a_stock_eod_price.amount
FROM ts_a_stock_eod_price, 
	(
		select distinct(link_symbol) as w_missing_symbol, w_symbol from ts_link_table 
		WHERE adj_ratio is NULL
	) missing_table
WHERE ts_a_stock_eod_price.symbol = missing_table.w_missing_symbol;

/* Set new stock adj ratio to 1 */
UPDATE ts_link_table  SET adj_ratio=1 WHERE adj_ratio is NULL;

/* Fill in index price from ts */
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
WHERE tradedate > 
		(
			select max(tradedate) as tradedate
			FROM final_a_stock_comb_info 
			where symbol = "SZ399300"
		) 
) ts_raw_table
LEFT JOIN ts_link_table ON ts_raw_table.symbol = ts_link_table.link_symbol;

/* Fill in stock price from ts */
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
WHERE tradedate > (
	select max(tradedate) as tradedate
	FROM
		(select tradedate, count(tradedate) as symbol_count 
		FROM final_a_stock_comb_info 
		where tradedate > "2022-07-01" 
		group by tradedate) tradedate_record
	WHERE symbol_count > 1000
  )
) ts_raw_table
LEFT JOIN ts_link_table ON ts_raw_table.symbol = ts_link_table.link_symbol;

/* First, identify and print records from ts_a_stock_fundamental that do not exist in final_a_stock_comb_info */
SELECT 
  CONCAT('Missing record: tradedate=', STR_TO_DATE(ts_raw.trade_date, '%Y%m%d'), ', symbol=', ts_link_table.w_symbol) AS missing_info
FROM ts_a_stock_fundamental ts_raw
LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
LEFT JOIN final_a_stock_comb_info final ON STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') = final.tradedate AND ts_link_table.w_symbol = final.symbol
WHERE STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') > COALESCE((SELECT MAX(tradedate) FROM final_a_stock_comb_info), '2008-01-01')
AND final.tradedate IS NULL;

/* Then, update existing records in final_a_stock_comb_info with data from ts_a_stock_fundamental */
/* Updated logic: Use TTM values for pe, ps, dv_ratio when available and valid */
UPDATE final_a_stock_comb_info final
INNER JOIN (
  SELECT
    STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') AS tradedate,
    ts_link_table.w_symbol AS symbol,
    ts_raw.turnover_rate_f / 100.0 AS turnover_rate,
    ts_raw.volume_ratio AS volume_ratio,
    -- Use pe_ttm if available and valid, otherwise use pe
    CASE
      WHEN ts_raw.pe_ttm IS NOT NULL THEN ts_raw.pe_ttm
      ELSE ts_raw.pe
    END AS pe_final,
    ts_raw.pe_ttm AS pe_ttm,
    ts_raw.pb AS pb,
    -- Use ps_ttm if available and valid, otherwise use ps
    CASE
      WHEN ts_raw.ps_ttm IS NOT NULL THEN ts_raw.ps_ttm
      ELSE ts_raw.ps
    END AS ps_final,
    ts_raw.ps_ttm AS ps_ttm,
    -- Use dv_ttm if available and valid, otherwise use dv_ratio
    CASE
      WHEN ts_raw.dv_ttm IS NOT NULL THEN ts_raw.dv_ttm / 100.0
      ELSE ts_raw.dv_ratio / 100.0
    END AS dv_ratio_final,
    ts_raw.dv_ttm / 100.0 AS dv_ttm,
    ts_raw.circ_mv * 10000.0 AS circ_mv
  FROM ts_a_stock_fundamental ts_raw
  LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
  WHERE STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') > COALESCE((SELECT MAX(tradedate) FROM final_a_stock_comb_info), '2008-01-01')
) AS updates ON final.tradedate = updates.tradedate AND final.symbol = updates.symbol
SET
  final.turnover_rate = updates.turnover_rate,
  final.volume_ratio = updates.volume_ratio,
  final.pe = updates.pe_final,  -- Use TTM value when available
  final.pb = updates.pb,
  final.ps = updates.ps_final,  -- Use TTM value when available
  final.dv_ratio = updates.dv_ratio_final,  -- Use TTM value when available
  final.circ_mv = updates.circ_mv;

/* First, identify and print records from ts_a_stock_moneyflow that do not exist in final_a_stock_comb_info */
SELECT 
  CONCAT('Missing record: tradedate=', STR_TO_DATE(ts_raw.trade_date, '%Y%m%d'), ', symbol=', ts_link_table.w_symbol) AS missing_info
FROM ts_a_stock_moneyflow ts_raw
LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
LEFT JOIN final_a_stock_comb_info final ON STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') = final.tradedate AND ts_link_table.w_symbol = final.symbol
WHERE STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') > COALESCE((SELECT MAX(tradedate) FROM final_a_stock_comb_info), '2008-01-01')
AND final.tradedate IS NULL;

/* Then, update existing records in final_a_stock_comb_info with data from ts_a_stock_moneyflow */
UPDATE final_a_stock_comb_info final
INNER JOIN (
  SELECT
    STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') AS tradedate,
    ts_link_table.w_symbol AS symbol,
    CASE
      WHEN (ts_raw.buy_sm_amount + ts_raw.buy_md_amount + ts_raw.buy_lg_amount + ts_raw.buy_elg_amount) > 0
      THEN ((ts_raw.buy_lg_amount + ts_raw.buy_elg_amount) - (ts_raw.sell_lg_amount + ts_raw.sell_elg_amount)) / (ts_raw.buy_sm_amount + ts_raw.buy_md_amount + ts_raw.buy_lg_amount + ts_raw.buy_elg_amount)
      ELSE 0
    END AS main_inflow_ratio,
    CASE
      WHEN (ts_raw.buy_sm_amount + ts_raw.buy_md_amount + ts_raw.buy_lg_amount + ts_raw.buy_elg_amount) > 0
      THEN ((ts_raw.buy_sm_amount + ts_raw.buy_md_amount) - (ts_raw.sell_sm_amount + ts_raw.sell_md_amount)) / (ts_raw.buy_sm_amount + ts_raw.buy_md_amount + ts_raw.buy_lg_amount + ts_raw.buy_elg_amount)
      ELSE 0
    END AS small_inflow_ratio,
    CASE
      WHEN (ts_raw.buy_sm_amount + ts_raw.buy_md_amount + ts_raw.buy_lg_amount + ts_raw.buy_elg_amount) > 0
      THEN ts_raw.net_mf_amount / (ts_raw.buy_sm_amount + ts_raw.buy_md_amount + ts_raw.buy_lg_amount + ts_raw.buy_elg_amount)
      ELSE 0
    END AS net_inflow_ratio
  FROM ts_a_stock_moneyflow ts_raw
  LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
  WHERE STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') > COALESCE((SELECT MAX(tradedate) FROM final_a_stock_comb_info), '2008-01-01')
) AS updates ON final.tradedate = updates.tradedate AND final.symbol = updates.symbol
SET
  final.main_inflow_ratio = updates.main_inflow_ratio,
  final.small_inflow_ratio = updates.small_inflow_ratio,
  final.net_inflow_ratio = updates.net_inflow_ratio;

/* First, identify and print records from ts_a_stock_cost_pct that do not exist in final_a_stock_comb_info */
SELECT
  CONCAT('Missing record: tradedate=', STR_TO_DATE(ts_raw.trade_date, '%Y%m%d'), ', symbol=', ts_link_table.w_symbol) AS missing_info
FROM ts_a_stock_cost_pct ts_raw
LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
LEFT JOIN final_a_stock_comb_info final ON STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') = final.tradedate AND ts_link_table.w_symbol = final.symbol
WHERE STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') > COALESCE((SELECT MAX(tradedate) FROM final_a_stock_comb_info), '2008-01-01')
AND final.tradedate IS NULL;

/* Then, update existing records in final_a_stock_comb_info with data from ts_a_stock_cost_pct */
UPDATE final_a_stock_comb_info final
INNER JOIN (
  SELECT
    STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') AS tradedate,
    ts_link_table.w_symbol AS symbol,
    ts_raw.cost_5pct AS cost_5pct,
    ts_raw.cost_15pct AS cost_15pct,
    ts_raw.cost_50pct AS cost_50pct,
    ts_raw.cost_85pct AS cost_85pct,
    ts_raw.cost_95pct AS cost_95pct,
    ts_raw.weight_avg AS weight_avg,
    ts_raw.winner_rate AS winner_rate
  FROM ts_a_stock_cost_pct ts_raw
  LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
  WHERE STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') > COALESCE((SELECT MAX(tradedate) FROM final_a_stock_comb_info), '2008-01-01')
) AS updates ON final.tradedate = updates.tradedate AND final.symbol = updates.symbol
SET
  final.cost_5pct = updates.cost_5pct,
  final.cost_15pct = updates.cost_15pct,
  final.cost_50pct = updates.cost_50pct,
  final.cost_85pct = updates.cost_85pct,
  final.cost_95pct = updates.cost_95pct,
  final.weight_avg = updates.weight_avg,
  final.winner_rate = updates.winner_rate;

/* First, identify and print records from ts_a_stock_suspend_info that do not exist in final_a_stock_comb_info */
SELECT
  CONCAT('Missing record: tradedate=', STR_TO_DATE(ts_raw.trade_date, '%Y%m%d'), ', symbol=', ts_link_table.w_symbol) AS missing_info
FROM ts_a_stock_suspend_info ts_raw
LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
LEFT JOIN final_a_stock_comb_info final ON STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') = final.tradedate AND ts_link_table.w_symbol = final.symbol
WHERE STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') > COALESCE((SELECT MAX(tradedate) FROM final_a_stock_comb_info), '2008-01-01')
AND final.tradedate IS NULL;

/* Then, update existing records in final_a_stock_comb_info with data from ts_a_stock_suspend_info */
UPDATE final_a_stock_comb_info final
INNER JOIN (
  SELECT
    STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') AS tradedate,
    ts_link_table.w_symbol AS symbol,
    CASE
      WHEN ts_raw.suspend_type = 'S' THEN TRUE
      ELSE FALSE
    END AS suspend
  FROM ts_a_stock_suspend_info ts_raw
  LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
  WHERE STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') > COALESCE((SELECT MAX(tradedate) FROM final_a_stock_comb_info), '2008-01-01')
) AS updates ON final.tradedate = updates.tradedate AND final.symbol = updates.symbol
SET
  final.suspend = updates.suspend;

/* Update final_a_stock_comb_info with consensus report data - use min_price as target_price */
UPDATE final_a_stock_comb_info final
INNER JOIN (
  SELECT
    STR_TO_DATE(consensus.eval_date, '%Y%m%d') AS tradedate,
    ts_link_table.w_symbol AS symbol,
    consensus.total_reports,
    consensus.sentiment_pos,
    consensus.sentiment_neg,
    consensus.eps AS f_eps,
    consensus.pe AS f_pe,
    consensus.rd AS f_dv_ratio,  -- Map rd (dividend ratio) to f_dv_ratio
    consensus.roe AS f_roe,
    consensus.min_price AS f_target_price,  -- Use min_price as target_price as requested
  FROM ts_a_stock_consensus_report consensus
  LEFT JOIN ts_link_table ON consensus.ts_code = ts_link_table.link_symbol
  WHERE STR_TO_DATE(consensus.eval_date, '%Y%m%d') > COALESCE((SELECT MAX(tradedate) FROM final_a_stock_comb_info), '2008-01-01')
    AND consensus.report_period LIKE '%2025%'  -- Focus on current year data
    AND consensus.total_reports > 0  -- Only include records with actual reports
) AS consensus_updates ON final.tradedate = consensus_updates.tradedate AND final.symbol = consensus_updates.symbol
SET
  final.f_pos_ratio = CASE
    WHEN consensus_updates.total_reports > 0 THEN consensus_updates.sentiment_pos / consensus_updates.total_reports
    ELSE 0
  END,
  final.f_neg_ratio = CASE
    WHEN consensus_updates.total_reports > 0 THEN consensus_updates.sentiment_neg / consensus_updates.total_reports
    ELSE 0
  END,
  final.f_eps = consensus_updates.f_eps,
  final.f_pe = consensus_updates.f_pe,
  final.f_dv_ratio = consensus_updates.f_dv_ratio,
  final.f_roe = consensus_updates.f_roe,
  final.f_target_price = consensus_updates.f_target_price;

