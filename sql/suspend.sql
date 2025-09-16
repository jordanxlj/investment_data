/* Module 11: Update Suspend Info - Identify Missing and Update */

SELECT "Identify and print records from ts_a_stock_suspend_info that do not exist in final_a_stock_comb_info" as info;
SELECT
  CONCAT('Missing suspend record: tradedate=', ts_raw.trade_date, ', symbol=', ts_link_table.w_symbol) AS missing_info
FROM ts_a_stock_suspend_info ts_raw
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
    CASE
      WHEN ts_raw.suspend_type = 'S' THEN TRUE
      ELSE FALSE
    END AS suspend
  FROM ts_a_stock_suspend_info ts_raw
  LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
  WHERE ts_raw.trade_date >= @start_date
    AND ts_raw.trade_date > @max_tradedate
) AS updates ON final.tradedate = updates.tradedate AND final.symbol = updates.symbol
SET
  final.suspend = updates.suspend;

/* Debug: Count updated rows and suspend count */
SELECT ROW_COUNT() AS updated_count;
IF @debug = 1 THEN
    SELECT COUNT(*) AS suspend_true FROM final_a_stock_comb_info WHERE suspend = TRUE;
END IF;