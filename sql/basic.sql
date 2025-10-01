/* ============================================================================
   Basic Information Update Script

   This script updates the industry field in final_a_stock_comb_info table
   by fetching industry_code from ts_a_stock_basic and converting it to INT.

   It also records the update in update_record_table.

   ============================================================================ */

SET @start_date = '2013-01-01';  /* Start date for data processing */
SET @debug = 0;  /* Set to 1 to enable debug output */

SELECT CONCAT('Basic Info Update: Processing data from: ', @start_date, ', debug = ', @debug) AS update_info;

/* Get the last basic_industry update time from the tracking table */
SET @update_start = COALESCE(
    (SELECT MAX(end_day) FROM update_record_table
     WHERE update_type = 'basic_industry'),
    @start_date
);

SELECT CONCAT('Last basic_industry update was on: ', @update_start) AS last_update_info;

/* Step 1: Update industry field in final_a_stock_comb_info */
SELECT "Updating industry field in final_a_stock_comb_info..." as info;

UPDATE final_a_stock_comb_info target
INNER JOIN ts_link_table link ON target.symbol = link.w_symbol
INNER JOIN ts_a_stock_basic basic ON basic.ts_code = link.link_symbol
SET target.industry = CAST(basic.industry_code AS SIGNED INTEGER)
WHERE basic.industry_code IS NOT NULL
  AND basic.industry_code REGEXP '^[0-9]+$'  /* Only update if industry_code is numeric */
  AND target.tradedate > @update_start;  /* Only update records after last update */

SET @updated_records = ROW_COUNT();

/* Step 2: Verify update */
SELECT "Update verification:" as info;
SELECT
    @updated_records as updated_records,
    (SELECT COUNT(*) FROM final_a_stock_comb_info WHERE industry IS NOT NULL) as total_industry_records,
    (SELECT COUNT(*) FROM final_a_stock_comb_info) as total_records;

/* Show sample of updated data */
SELECT "Sample of updated industry data:" as info;
SELECT
    symbol,
    tradedate,
    industry,
    close
FROM final_a_stock_comb_info
WHERE industry IS NOT NULL
ORDER BY tradedate DESC, symbol
LIMIT 10;

/* Step 3: Calculate actual statistics and record the update in update_record_table */
SET @actual_max_date = (SELECT MAX(tradedate) FROM final_a_stock_comb_info
                        WHERE industry IS NOT NULL);

INSERT INTO update_record_table (
    update_type, end_day, start_day, record_count, last_update_time
) VALUES (
    'basic_industry',
    @actual_max_date,
    @update_start,
    @updated_records,
    NOW()
);

SELECT "Industry update completed successfully!" as info;
