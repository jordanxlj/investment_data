/* Module 10: Update Cost Pct - Identify Missing and Update (Day by Day) */
SET @start_date = '2025-09-10';  /* Default start date */
SET @debug = 0;

/* Get the last cost update time from the tracking table */
SET @last_cost_update = COALESCE(
    (SELECT MAX(end_day) FROM update_record_table
     WHERE update_type = 'cost_pct'),
    @start_date
);

SELECT CONCAT('Last cost update was on: ', @last_cost_update) AS last_update_info;

SELECT "Identify and print records from ts_a_stock_cost_pct that do not exist in final_a_stock_comb_info" as info;
SELECT
  CONCAT('Missing cost record: tradedate=', ts_raw.trade_date, ', symbol=', ts_link_table.w_symbol) AS missing_info
FROM ts_a_stock_cost_pct ts_raw
LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
LEFT JOIN final_a_stock_comb_info final ON ts_raw.trade_date = final.tradedate AND ts_link_table.w_symbol = final.symbol
WHERE ts_raw.trade_date >= @last_cost_update
  AND @debug = 1  /* Only show debug info when debug is enabled */
AND final.tradedate IS NULL;

/* Debug: Check source data availability */
SELECT MAX(trade_date) AS max_source_date, COUNT(*) AS source_rows FROM ts_a_stock_cost_pct WHERE trade_date > @last_cost_update;

SELECT "Update existing records in final_a_stock_comb_info with data from ts_a_stock_cost_pct (day by day)" as info;

/* Create a temporary table to store dates to process */
DROP TEMPORARY TABLE IF EXISTS temp_dates_to_process;
CREATE TEMPORARY TABLE temp_dates_to_process AS
SELECT DISTINCT trade_date
FROM ts_a_stock_cost_pct
WHERE trade_date >= @last_cost_update
ORDER BY trade_date;

/* Debug: Check if temp table has rows (reason for loop not entering) */
SELECT COUNT(*) AS dates_to_process FROM temp_dates_to_process;

/* Process each date individually using a stored procedure */
-- Change delimiter to handle multi-statement procedure
DELIMITER //

-- Create a stored procedure to encapsulate the loop with optimized join
DROP PROCEDURE IF EXISTS process_cost_pct_batch;
CREATE PROCEDURE process_cost_pct_batch()
BEGIN
  DECLARE v_current_date DATE;
  DECLARE processed_count INT DEFAULT 0;
  DECLARE total_updated INT DEFAULT 0;
  DECLARE done INT DEFAULT FALSE;
  DECLARE date_cursor CURSOR FOR
    SELECT trade_date
    FROM temp_dates_to_process
    ORDER BY trade_date;
  DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;

  -- Create temporary table with pre-joined data for better performance
  DROP TEMPORARY TABLE IF EXISTS temp_cost_pct_joined;
  CREATE TEMPORARY TABLE temp_cost_pct_joined AS
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
  WHERE ts_raw.trade_date >= @last_cost_update;

  -- Create index on the temporary table for better performance
  CREATE INDEX idx_temp_cost_pct_date ON temp_cost_pct_joined (tradedate);

  /* Debug: Check joined temp table rows */
  SELECT COUNT(*) AS joined_rows FROM temp_cost_pct_joined;

  -- Open the cursor
  OPEN date_cursor;

  read_loop: LOOP
    FETCH date_cursor INTO v_current_date;
    IF done THEN
      LEAVE read_loop;
    END IF;

    SELECT CONCAT('Update Cost Pct, Processing date: ', v_current_date) as processing_info;

    /* Update records for this specific date using pre-joined temp table */
    UPDATE final_a_stock_comb_info final
    INNER JOIN temp_cost_pct_joined updates ON final.tradedate = updates.tradedate
                                           AND final.symbol = updates.symbol
    SET
      final.cost_5pct = updates.cost_5pct,
      final.cost_15pct = updates.cost_15pct,
      final.cost_50pct = updates.cost_50pct,
      final.cost_85pct = updates.cost_85pct,
      final.cost_95pct = updates.cost_95pct,
      final.weight_avg = updates.weight_avg,
      final.winner_rate = updates.winner_rate
    WHERE updates.tradedate = v_current_date;

    /* Get rows updated for this date and accumulate */
    SET @updated_for_date = ROW_COUNT();
    SET total_updated = total_updated + @updated_for_date;

    /* Debug: Rows updated for this date */
    SELECT @updated_for_date AS updated_for_date;

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

  -- Record the update in tracking table
  IF total_updated > 0 THEN
    INSERT INTO update_record_table (
      update_type, end_day, start_day, record_count, last_update_time
    ) VALUES (
      'cost_pct',
      v_current_date,  -- Last processed date
      @last_cost_update,
      total_updated,
      NOW()
    );
  END IF;

  -- Clean up the temporary table
  DROP TEMPORARY TABLE IF EXISTS temp_cost_pct_joined;
END //

-- Reset delimiter
DELIMITER ;

-- Call the procedure to execute
CALL process_cost_pct_batch();

/* Debug: Total processed after call */
SELECT 'Debug: Procedure completed' AS status;

-- Clean up the procedure after use
DROP PROCEDURE IF EXISTS process_cost_pct_batch;
/* Clean up temporary table */
DROP TEMPORARY TABLE IF EXISTS temp_dates_to_process;
