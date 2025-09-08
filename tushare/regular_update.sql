/* Create final table for combined info if it does not exist
   - percentages/ratios stored as FLOAT (already divided by 100 in SELECT)
   - shares/market cap stored in base units (×10000), as BIGINT UNSIGNED */
CREATE TABLE IF NOT EXISTS final_a_stock_comb_info (
  tradedate DATE,
  symbol VARCHAR(16),
  high FLOAT,
  low FLOAT,
  open FLOAT,
  close FLOAT,
  volume BIGINT UNSIGNED,
  adjclose FLOAT,
  amount BIGINT UNSIGNED,
  turnover_rate FLOAT,
  turnover_rate_f FLOAT,
  volume_ratio FLOAT,
  pe FLOAT,
  pe_ttm FLOAT,
  pb FLOAT,
  ps FLOAT,
  ps_ttm FLOAT,
  dv_ratio FLOAT,
  dv_ttm FLOAT,
  total_share BIGINT UNSIGNED,
  float_share BIGINT UNSIGNED,
  free_share BIGINT UNSIGNED,
  total_mv BIGINT UNSIGNED,
  circ_mv BIGINT UNSIGNED,
  main_inflow_ratio FLOAT,
  small_inflow_ratio FLOAT,
  net_inflow_ratio FLOAT,
  PRIMARY KEY (tradedate, symbol),
  KEY idx_comb_symbol_tradedate (symbol, tradedate)
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
UPDATE final_a_stock_comb_info final
INNER JOIN (
  SELECT 
    STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') AS tradedate,
    ts_link_table.w_symbol AS symbol,
    ts_raw.turnover_rate / 100.0 AS turnover_rate,
    ts_raw.turnover_rate_f / 100.0 AS turnover_rate_f,
    ts_raw.volume_ratio AS volume_ratio,
    ts_raw.pe AS pe,
    ts_raw.pe_ttm AS pe_ttm,
    ts_raw.pb AS pb,
    ts_raw.ps AS ps,
    ts_raw.ps_ttm AS ps_ttm,
    ts_raw.dv_ratio / 100.0 AS dv_ratio,
    ts_raw.dv_ttm / 100.0 AS dv_ttm,
    ts_raw.total_share * 10000.0 AS total_share,
    ts_raw.float_share * 10000.0 AS float_share,
    ts_raw.free_share * 10000.0 AS free_share,
    ts_raw.total_mv * 10000.0 AS total_mv,
    ts_raw.circ_mv * 10000.0 AS circ_mv
  FROM ts_a_stock_fundamental ts_raw
  LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
  WHERE STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') > COALESCE((SELECT MAX(tradedate) FROM final_a_stock_comb_info), '2008-01-01')
) AS updates ON final.tradedate = updates.tradedate AND final.symbol = updates.symbol
SET
  final.turnover_rate = updates.turnover_rate,
  final.turnover_rate_f = updates.turnover_rate_f,
  final.volume_ratio = updates.volume_ratio,
  final.pe = updates.pe,
  final.pe_ttm = updates.pe_ttm,
  final.pb = updates.pb,
  final.ps = updates.ps,
  final.ps_ttm = updates.ps_ttm,
  final.dv_ratio = updates.dv_ratio,
  final.dv_ttm = updates.dv_ttm,
  final.total_share = updates.total_share,
  final.float_share = updates.float_share,
  final.free_share = updates.free_share,
  final.total_mv = updates.total_mv,
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
      THEN (ts_raw.buy_sm_amount - ts_raw.sell_sm_amount) / (ts_raw.buy_sm_amount + ts_raw.buy_md_amount + ts_raw.buy_lg_amount + ts_raw.buy_elg_amount)
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