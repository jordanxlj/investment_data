/* Module 9: Update Moneyflow - Identify Missing and Update (Day by Day) */
SET @max_tradedate = (SELECT COALESCE(MAX(tradedate), '2010l-01-01') FROM final_a_stock_comb_info);
SET @start_date = '2025-09-01';
SET @debug = 0;

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

/* Debug: Check source data availability */
SELECT MAX(trade_date) AS max_source_date, COUNT(*) AS source_rows FROM ts_a_stock_moneyflow WHERE trade_date > @max_tradedate;

SELECT "Update existing records in final_a_stock_comb_info with data from ts_a_stock_moneyflow (day by day)" as info;

/* Create a temporary table to store dates to process */
DROP TEMPORARY TABLE IF EXISTS temp_dates_to_process;
CREATE TEMPORARY TABLE temp_dates_to_process AS
SELECT DISTINCT trade_date
FROM ts_a_stock_moneyflow
WHERE trade_date >= @start_date
  AND trade_date > @max_tradedate
ORDER BY trade_date;

/* Debug: Check if temp table has rows (reason for loop not entering) */
SELECT COUNT(*) AS dates_to_process FROM temp_dates_to_process;

/* Process each date individually using a stored procedure */
-- Change delimiter to handle multi-statement procedure
DELIMITER //

-- Create a stored procedure to encapsulate the loop with optimized join
DROP PROCEDURE IF EXISTS process_moneyflow_batch;
CREATE PROCEDURE process_moneyflow_batch()
BEGIN
  DECLARE v_current_date DATE;
  DECLARE processed_count INT DEFAULT 0;
  DECLARE done INT DEFAULT FALSE;
  DECLARE date_cursor CURSOR FOR
    SELECT trade_date
    FROM temp_dates_to_process
    ORDER BY trade_date;
  DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;

  -- Create temporary table with pre-joined data and pre-calculated ratios for better performance
  DROP TEMPORARY TABLE IF EXISTS temp_moneyflow_joined;
  CREATE TEMPORARY TABLE temp_moneyflow_joined AS
  SELECT
    ts_raw.trade_date AS tradedate,
    ts_link_table.w_symbol AS symbol,
    -- Pre-calculate ratios to avoid complex calculations in UPDATE
    CASE
      WHEN (ts_raw.buy_sm_amount + ts_raw.buy_md_amount + ts_raw.buy_lg_amount + ts_raw.buy_elg_amount) > 0
      THEN ((ts_raw.buy_lg_amount + ts_raw.buy_elg_amount) - (ts_raw.sell_lg_amount + ts_raw.sell_elg_amount)) / (ts_raw.buy_sm_amount + ts_raw.buy_md_amount + ts_raw.buy_lg_amount + ts_raw.buy_elg_amount) * 100.0
      ELSE 0
    END AS main_inflow_ratio,
    CASE
      WHEN (ts_raw.buy_sm_amount + ts_raw.buy_md_amount + ts_raw.buy_lg_amount + ts_raw.buy_elg_amount) > 0
      THEN ((ts_raw.buy_sm_amount + ts_raw.buy_md_amount) - (ts_raw.sell_sm_amount + ts_raw.sell_md_amount)) / (ts_raw.buy_sm_amount + ts_raw.buy_md_amount + ts_raw.buy_lg_amount + ts_raw.buy_elg_amount) * 100.0
      ELSE 0
    END AS small_inflow_ratio,
    CASE
      WHEN (ts_raw.buy_sm_amount + ts_raw.buy_md_amount + ts_raw.buy_lg_amount + ts_raw.buy_elg_amount) > 0
      THEN ts_raw.net_mf_amount / (ts_raw.buy_sm_amount + ts_raw.buy_md_amount + ts_raw.buy_lg_amount + ts_raw.buy_elg_amount) * 100.0
      ELSE 0
    END AS net_inflow_ratio,
    -- Store denominator for potential future use
    (ts_raw.buy_sm_amount + ts_raw.buy_md_amount + ts_raw.buy_lg_amount + ts_raw.buy_elg_amount) AS total_buy_amount
  FROM ts_a_stock_moneyflow ts_raw
  LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
  WHERE ts_raw.trade_date >= @start_date
    AND ts_raw.trade_date > @max_tradedate;

  -- Create index on the temporary table for better performance
  CREATE INDEX idx_temp_moneyflow_date ON temp_moneyflow_joined (tradedate);

  /* Debug: Check joined temp table rows */
  SELECT COUNT(*) AS joined_rows FROM temp_moneyflow_joined;

  -- Open the cursor
  OPEN date_cursor;

  read_loop: LOOP
    FETCH date_cursor INTO v_current_date;
    IF done THEN
      LEAVE read_loop;
    END IF;

    SELECT CONCAT('Update Moneyflow, Processing date: ', v_current_date) as processing_info;

    /* Update records for this specific date using pre-joined temp table */
    UPDATE final_a_stock_comb_info final
    INNER JOIN temp_moneyflow_joined updates ON final.tradedate = updates.tradedate
                                           AND final.symbol = updates.symbol
    SET
      final.main_inflow_ratio = updates.main_inflow_ratio,
      final.small_inflow_ratio = updates.small_inflow_ratio,
      final.net_inflow_ratio = updates.net_inflow_ratio
    WHERE updates.tradedate = v_current_date;

    /* Debug: Rows updated for this date */
    SELECT ROW_COUNT() AS updated_for_date;

    /* Remove processed date from temp table */
    DELETE FROM temp_dates_to_process WHERE trade_date = v_current_date;

    /* Increment counter */
    SET processed_count = processed_count + 1;

    /* Add a small delay every 10 dates to prevent overwhelming the server */
    IF processed_count % 10 = 0 THEN
      DO SLEEP(0.1);
    END IF;

  END LOOP read_loop;

  CLOSE date_cursor;

  -- Clean up the temporary table
  DROP TEMPORARY TABLE IF EXISTS temp_moneyflow_joined;
END //

-- Reset delimiter
DELIMITER ;

-- Call the procedure to execute
CALL process_moneyflow_batch();

/* Debug: Total processed after call */
SELECT 'Debug: Procedure completed' AS status;

-- Clean up the procedure after use
DROP PROCEDURE IF EXISTS process_moneyflow_batch;
/* Clean up temporary table */
DROP TEMPORARY TABLE IF EXISTS temp_dates_to_process;
