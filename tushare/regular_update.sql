/* Add new stock to ts_link_table */
INSERT IGNORE INTO ts_link_table (w_symbol, link_symbol, link_date)
select concat(substr(symbol, 8, 2), substr(symbol, 1, 6)) as w_symbol, symbol as link_symbol, max(tradedate) as link_date 
from ts_a_stock_eod_price 
where tradedate = (select max(tradedate) from ts_a_stock_eod_price) group by symbol;

/* Fill in new stock price */
/* Fill in stock where w stock does not exists */
INSERT IGNORE INTO final_a_stock_eod_price (tradedate, symbol, high, low, open, close, volume, adjclose, amount)
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
INSERT IGNORE INTO final_a_stock_eod_price (tradedate, symbol, high, low, open, close, volume, adjclose, amount) 
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
			FROM final_a_stock_eod_price 
			where symbol = "SZ399300"
		) 
) ts_raw_table
LEFT JOIN ts_link_table ON ts_raw_table.symbol = ts_link_table.link_symbol;

/* ========================= */
/* Fill in fundamentals from ts */
/* ========================= */

/* Create final table for fundamentals if it does not exist
   - percentages/ratios stored as FLOAT (already divided by 100 in SELECT)
   - shares/market cap stored in base units (×10000), as BIGINT UNSIGNED */
CREATE TABLE IF NOT EXISTS final_a_stock_fundamental (
  tradedate DATE,
  symbol VARCHAR(16),
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
  PRIMARY KEY (tradedate, symbol)
);

/* Append newly imported ts fundamentals into final table */
INSERT IGNORE INTO final_a_stock_fundamental (
  tradedate, symbol, turnover_rate, turnover_rate_f, volume_ratio,
  pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm,
  total_share, float_share, free_share, total_mv, circ_mv
)
SELECT 
  STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') AS tradedate,
  ts_link_table.w_symbol AS symbol,
  ts_raw.turnover_rate / 100.0,
  ts_raw.turnover_rate_f / 100.0,
  ts_raw.volume_ratio,
  ts_raw.pe,
  ts_raw.pe_ttm,
  ts_raw.pb,
  ts_raw.ps,
  ts_raw.ps_ttm,
  ts_raw.dv_ratio / 100.0,
  ts_raw.dv_ttm / 100.0,
  ts_raw.total_share * 10000.0,
  ts_raw.float_share * 10000.0,
  ts_raw.free_share * 10000.0,
  ts_raw.total_mv * 10000.0,
  ts_raw.circ_mv * 10000.0
FROM ts_a_stock_fundamental ts_raw
LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
WHERE STR_TO_DATE(ts_raw.trade_date, '%Y%m%d') > COALESCE((SELECT MAX(tradedate) FROM final_a_stock_fundamental), '2008-01-01');

/* Fill in stock price from ts */
INSERT IGNORE INTO final_a_stock_eod_price (tradedate, symbol, high, low, open, close, volume, adjclose, amount) 
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
		FROM final_a_stock_eod_price 
		where tradedate > "2022-07-01" 
		group by tradedate) tradedate_record
	WHERE symbol_count > 1000
  )
) ts_raw_table
LEFT JOIN ts_link_table ON ts_raw_table.symbol = ts_link_table.link_symbol;