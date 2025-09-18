/* Update Fundamentals - Identify Missing and Update (Day by Day) */
SET @max_tradedate = (SELECT COALESCE(MAX(tradedate), '2008-01-01') FROM final_a_stock_comb_info);
SET @start_date = '2025-09-01';
SET @debug = 1;  -- Enable debug for more outputs

SELECT "Identify and print records from ts_a_stock_fundamental that do not exist in final_a_stock_comb_info" as info;
SELECT
  CONCAT('Missing fundamental record: tradedate=', ts_raw.trade_date, ', symbol=', ts_link_table.w_symbol) AS missing_info
FROM ts_a_stock_fundamental ts_raw
LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
LEFT JOIN final_a_stock_comb_info final ON ts_raw.trade_date = final.tradedate AND ts_link_table.w_symbol = final.symbol
WHERE ts_raw.trade_date >= @start_date
  AND ts_raw.trade_date > @max_tradedate
  AND @debug = 1  /* Only show debug info when debug is enabled */
AND final.tradedate IS NULL;

/* Debug: Check source data availability */
SELECT MAX(trade_date) AS max_source_date, COUNT(*) AS source_rows FROM ts_a_stock_fundamental WHERE trade_date > @max_tradedate;

SELECT "Update existing records in final_a_stock_comb_info with data from ts_a_stock_fundamental (day by day)" as info;

/* Create a temporary table to store dates to process */
DROP TEMPORARY TABLE IF EXISTS temp_dates_to_process;
CREATE TEMPORARY TABLE temp_dates_to_process AS
SELECT DISTINCT trade_date
FROM ts_a_stock_fundamental
WHERE trade_date >= @start_date
  AND trade_date > @max_tradedate
ORDER BY trade_date;

/* Debug: Check if temp table has rows (reason for loop not entering) */
SELECT COUNT(*) AS dates_to_process FROM temp_dates_to_process;

/* Process each date individually using a stored procedure */
-- Change delimiter to handle multi-statement procedure
DELIMITER //

-- Create a stored procedure to encapsulate the loop with optimized join
DROP PROCEDURE IF EXISTS process_fundamentals_batch;
CREATE PROCEDURE process_fundamentals_batch()
BEGIN
  DECLARE v_current_date DATE;
  DECLARE processed_count INT DEFAULT 0;
  DECLARE done INT DEFAULT FALSE;
  DECLARE date_cursor CURSOR FOR
    SELECT trade_date
    FROM temp_dates_to_process
    ORDER BY trade_date;
  DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;

  -- Create temporary table with pre-joined data for better performance
  DROP TEMPORARY TABLE IF EXISTS temp_fundamentals_joined;
  CREATE TEMPORARY TABLE temp_fundamentals_joined AS
  SELECT
    ts_raw.trade_date AS tradedate,
    ts_link_table.w_symbol AS symbol,
    ts_raw.turnover_rate_f AS turnover_rate,
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
      WHEN ts_raw.dv_ttm IS NOT NULL THEN ts_raw.dv_ttm
      ELSE ts_raw.dv_ratio
    END AS dv_ratio_final,
    ts_raw.dv_ttm AS dv_ttm,
    ts_raw.circ_mv * 10000.0 AS circ_mv
  FROM ts_a_stock_fundamental ts_raw
  LEFT JOIN ts_link_table ON ts_raw.ts_code = ts_link_table.link_symbol
  WHERE ts_raw.trade_date >= @start_date
    AND ts_raw.trade_date > @max_tradedate;

  -- Create index on the temporary table for better performance
  CREATE INDEX idx_temp_fundamentals_date ON temp_fundamentals_joined (tradedate);

  /* Debug: Check joined temp table rows */
  SELECT COUNT(*) AS joined_rows FROM temp_fundamentals_joined;

  -- Debug: Sample data from temp table for first date (if debug enabled)
  IF @debug = 1 THEN
    SELECT "Sample source data for first processing date" AS sample_info;
    SELECT tradedate, symbol, turnover_rate, pe_final, pb, ps_final, dv_ratio_final, circ_mv
    FROM temp_fundamentals_joined
    LIMIT 5;
  END IF;

  -- Open the cursor
  OPEN date_cursor;

  read_loop: LOOP
    FETCH date_cursor INTO v_current_date;
    IF done THEN
      LEAVE read_loop;
    END IF;

    SELECT CONCAT('Update Fundamental, Processing date: ', v_current_date) as processing_info;

    -- Debug: Count potential matches before UPDATE (rows that would be joined)
    SELECT 
      CONCAT('Debug: Potential matches for ', v_current_date, ': ', COUNT(*)) AS match_count_info
    FROM temp_fundamentals_joined updates
    INNER JOIN final_a_stock_comb_info final ON final.tradedate = updates.tradedate
                                             AND final.symbol = updates.symbol
    WHERE updates.tradedate = v_current_date;

    -- Debug: If debug enabled, sample existing vs source values for first symbol on this date
    IF @debug = 1 THEN
      SELECT "Sample comparison (existing vs source) for first symbol on this date" AS comparison_info;
      
      -- Sample symbol (pick the first one from final for this date)
      SELECT 
        CONCAT('final_symbol: ', final.symbol, ', turnover_rate: ', final.turnover_rate, 
               ', pe: ', final.pe, ', pb: ', final.pb, 
               ', ps: ', final.ps, ', dv_ratio: ', final.dv_ratio, 
               ', circ_mv: ', final.circ_mv) AS existing_values
      FROM final_a_stock_comb_info final
      WHERE final.tradedate = v_current_date
      LIMIT 1;
      
      SELECT 
        CONCAT('source_symbol: ', updates.symbol, ', turnover_rate: ', updates.turnover_rate, 
               ', pe_final: ', updates.pe_final, ', pb: ', updates.pb, 
               ', ps_final: ', updates.ps_final, ', dv_ratio_final: ', updates.dv_ratio_final, 
               ', circ_mv: ', updates.circ_mv) AS source_values
      FROM temp_fundamentals_joined updates
      WHERE updates.tradedate = v_current_date
      LIMIT 1;
    END IF;

    /* Update records for this specific date using pre-joined temp table */
    UPDATE final_a_stock_comb_info final
    INNER JOIN temp_fundamentals_joined updates ON final.tradedate = updates.tradedate
                                                 AND final.symbol = updates.symbol
    SET
      final.turnover_rate = updates.turnover_rate,
      final.volume_ratio = updates.volume_ratio,
      final.pe = updates.pe_final,  -- Use TTM value when available
      final.pb = updates.pb,
      final.ps = updates.ps_final,  -- Use TTM value when available
      final.dv_ratio = updates.dv_ratio_final,  -- Use TTM value when available
      final.circ_mv = updates.circ_mv
    WHERE updates.tradedate = v_current_date;

    /* Debug: Rows updated for this date (actual changes made) */
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
  DROP TEMPORARY TABLE IF EXISTS temp_fundamentals_joined;
END //

-- Reset delimiter
DELIMITER ;

-- Call the procedure to execute
CALL process_fundamentals_batch();

/* Debug: Total processed after call */
SELECT 'Debug: Procedure completed' AS status;

-- Clean up the procedure after use
DROP PROCEDURE IF EXISTS process_fundamentals_batch;
/* Clean up temporary table */
DROP TEMPORARY TABLE IF EXISTS temp_dates_to_process;