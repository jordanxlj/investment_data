/* Module 10: Update Cost Pct - Identify Missing and Update */

SELECT "Identify and print records from ts_a_stock_cost_pct that do not exist in final_a_stock_comb_info" as info;
SELECT
  CONCAT('Missing cost record: tradedate=', ts_raw.trade_date, ', symbol=', ts_link_table.w_symbol) AS missing_info
FROM ts_a_stock_cost_pct ts_raw
LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
LEFT JOIN final_a_stock_comb_info final ON ts_raw.trade_date = final.tradedate AND ts_link_table.w_symbol = final.symbol
WHERE ts_raw.trade_date >= @start_date
  AND ts_raw.trade_date > @max_tradedate
  AND @debug = 1  /* Only show debug info when debug is enabled */
AND final.tradedate IS NULL;

UPDATE final_a_stock_comb_info final
INNER JOIN (
  SELECT
    ts_raw.trade_date AS tradedate,
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
  WHERE ts_raw.trade_date >= @start_date
    AND ts_raw.trade_date > @max_tradedate
) AS updates ON final.tradedate = updates.tradedate AND final.symbol = updates.symbol
SET
  final.cost_5pct = updates.cost_5pct,
  final.cost_15pct = updates.cost_15pct,
  final.cost_50pct = updates.cost_50pct,
  final.cost_85pct = updates.cost_85pct,
  final.cost_95pct = updates.cost_95pct,
  final.weight_avg = updates.weight_avg,
  final.winner_rate = updates.winner_rate;

/* Debug: Count updated rows */
SELECT ROW_COUNT() AS updated_count;
IF @debug = 1 THEN
    SELECT * FROM final_a_stock_comb_info ORDER BY tradedate DESC LIMIT 1;
END IF;