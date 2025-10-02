/* Module 12: Update Consensus Report Data (Day by Day) */
SET @start_date = '2025-09-10';  /* Default start date */
SET @debug = 0;

/* Get the last brokerage report update time from the tracking table */
SET @last_brokerage_update = COALESCE(
    (SELECT MAX(end_day) FROM update_record_table
     WHERE update_type = 'brokerage_report'),
    @start_date
);

SELECT CONCAT('Last brokerage report update was on: ', @last_brokerage_update) AS last_update_info;

SELECT "Update final_a_stock_comb_info with consensus report data - use min_price as target_price (day by day)" as info;

/* Debug: Check source data availability */
SELECT MAX(eval_date) AS max_source_date, COUNT(*) AS source_rows
FROM ts_a_stock_consensus_report
WHERE eval_date >= @last_brokerage_update
  AND total_reports > 0;

/* Create a temporary table to store dates to process */
DROP TEMPORARY TABLE IF EXISTS temp_dates_to_process;
CREATE TEMPORARY TABLE temp_dates_to_process AS
SELECT DISTINCT eval_date AS trade_date
FROM ts_a_stock_consensus_report
WHERE eval_date >= @last_brokerage_update
  AND total_reports > 0
ORDER BY eval_date;

/* Debug: Check if temp table has rows (reason for loop not entering) */
SELECT COUNT(*) AS dates_to_process FROM temp_dates_to_process;

/* Process each date individually using a stored procedure */
-- Change delimiter to handle multi-statement procedure
DELIMITER //

-- Create a stored procedure to encapsulate the loop with optimized join
DROP PROCEDURE IF EXISTS process_consensus_batch;
CREATE PROCEDURE process_consensus_batch()
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
  DROP TEMPORARY TABLE IF EXISTS temp_consensus_joined;
  CREATE TEMPORARY TABLE temp_consensus_joined AS
  SELECT
    consensus.eval_date AS tradedate,
    ts_link_table.w_symbol AS symbol,
    consensus.total_reports,
    consensus.sentiment_pos,
    consensus.sentiment_neg,
    consensus.eps AS f_eps,
    consensus.pe AS f_pe,
    consensus.rd AS f_dv_ratio,  -- Map rd (dividend ratio) to f_dv_ratio
    consensus.roe AS f_roe,
    consensus.min_price AS f_target_price,  -- Use min_price as target_price as requested
    -- Pre-calculate ratios to avoid division in UPDATE
    CASE
      WHEN consensus.total_reports > 0 THEN consensus.sentiment_pos / consensus.total_reports * 100.0
      ELSE 0
    END AS f_pos_ratio,
    CASE
      WHEN consensus.total_reports > 0 THEN consensus.sentiment_neg / consensus.total_reports * 100.0
      ELSE 0
    END AS f_neg_ratio
  FROM ts_a_stock_consensus_report consensus
  LEFT JOIN ts_link_table ON consensus.ts_code = ts_link_table.link_symbol
  WHERE consensus.eval_date >= @last_brokerage_update
    AND consensus.total_reports > 0;

  -- Create index on the temporary table for better performance
  CREATE INDEX idx_temp_consensus_date ON temp_consensus_joined (tradedate);

  /* Debug: Check joined temp table rows */
  SELECT COUNT(*) AS joined_rows FROM temp_consensus_joined;

  -- Open the cursor
  OPEN date_cursor;

  read_loop: LOOP
    FETCH date_cursor INTO v_current_date;
    IF done THEN
      LEAVE read_loop;
    END IF;

    SELECT CONCAT('Update Consensus, Processing date: ', v_current_date) as processing_info;

    /* Update records for this specific date using pre-joined temp table */
    UPDATE final_a_stock_comb_info final
    INNER JOIN temp_consensus_joined updates ON final.tradedate = updates.tradedate
                                           AND final.symbol = updates.symbol
    SET
      final.f_pos_ratio = updates.f_pos_ratio,
      final.f_neg_ratio = updates.f_neg_ratio,
      final.f_eps = updates.f_eps,
      final.f_pe = updates.f_pe,
      final.f_dv_ratio = updates.f_dv_ratio,
      final.f_roe = updates.f_roe,
      final.f_target_price = updates.f_target_price
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
      'brokerage_report',
      v_current_date,  -- Last processed date
      @last_brokerage_update,
      total_updated,
      NOW()
    );
  END IF;

  -- Clean up the temporary table
  DROP TEMPORARY TABLE IF EXISTS temp_consensus_joined;
END //

-- Reset delimiter
DELIMITER ;

-- Call the procedure to execute
CALL process_consensus_batch();

/* Debug: Total processed after call */
SELECT 'Debug: Procedure completed' AS status;

-- Clean up the procedure after use
DROP PROCEDURE IF EXISTS process_consensus_batch;
/* Clean up temporary table */
DROP TEMPORARY TABLE IF EXISTS temp_dates_to_process;
