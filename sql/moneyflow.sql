/* Module 9: Update Moneyflow - Identify Missing and Update */

SELECT "Identify and print records from ts_a_stock_moneyflow that do not exist in final_a_stock_comb_info" as info;
SELECT 
  CONCAT('Missing moneyflow record: tradedate=', ts_raw.trade_date, ', symbol=', ts_link_table.w_symbol) AS missing_info
FROM ts_a_stock_moneyflow ts_raw
LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
LEFT JOIN final_a_stock_comb_info final ON ts_raw.trade_date = final.tradedate AND ts_link_table.w_symbol = final.symbol
WHERE ts_raw.trade_date >= @start_date
  AND ts_raw.trade_date > @max_tradedate
  AND @debug = 1  /* Only show debug info when debug is enabled */
AND final.tradedate IS NULL;

SELECT "Update existing records in final_a_stock_comb_info with data from ts_a_stock_moneyflow" as info;
UPDATE final_a_stock_comb_info final
INNER JOIN (
  SELECT
    ts_raw.trade_date AS tradedate,
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
  WHERE ts_raw.trade_date >= @start_date
    AND ts_raw.trade_date > @max_tradedate
) AS updates ON final.tradedate = updates.tradedate AND final.symbol = updates.symbol
SET
  final.main_inflow_ratio = updates.main_inflow_ratio,
  final.small_inflow_ratio = updates.small_inflow_ratio,
  final.net_inflow_ratio = updates.net_inflow_ratio;

/* Debug: Count updated rows */
SELECT ROW_COUNT() AS updated_count;
IF @debug = 1 THEN
    SELECT AVG(main_inflow_ratio) AS sample_avg FROM final_a_stock_comb_info LIMIT 1;
END IF;