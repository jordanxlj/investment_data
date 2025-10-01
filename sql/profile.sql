/* ============================================================================
   Financial Metrics Update Script for final_a_stock_comb_info table

   This script updates financial metrics from ts_a_stock_financial_profile
   to final_a_stock_comb_info based on announcement dates.

   FEATURES:
   - Batch processing by tradedate (configurable batch size, default 1 year)
   - Incremental updates based on last update time from update_record_table
   - Transactional batch processing with rollback on errors
   - Detailed progress tracking with individual batch records
   - Debug mode for monitoring batch processing
   - Proper DELIMITER handling for cursor blocks

   DEPENDENCY: Requires update_record_table to track last update times.

   Updated metrics:
   - Liquidity ratios (current_ratio, quick_ratio, cash_ratio)
   - Turnover ratios (ca_turn, inv_turn, ar_turn, fa_turn, assets_turn)
   - Basic profitability (roic, roe_ttm, roa_ttm, grossprofit_margin_ttm, netprofit_margin_ttm, fcf_margin_ttm)
   - Leverage ratios (debt_to_assets, debt_to_eqt, debt_to_ebitda)
   - Valuation metrics (bps, eps_ttm, revenue_ps_ttm, cfps, fcff_ps)
   - Growth metrics - YoY (or_yoy, netprofit_yoy, basic_eps_yoy, equity_yoy, assets_yoy, ocf_yoy, roe_yoy)
   - CAGR approximations (revenue_cagr_3y, netincome_cagr_3y)
   - Other (rd_exp_to_capex, goodwill)

   CONFIGURATION:
   - @batch_size_years: Number of years to process in each batch (default: 1)
   - @debug: Enable debug output (0=off, 1=on)

   ⚠️  IMPORTANT: TTM and CAGR values are approximations using latest available data.
   For accurate financial analysis, consider using the Python TTM calculation scripts.

   Run this after updating ts_a_stock_financial_profile data.

   ============================================================================ */

/* Set shared variables */
SET @start_date = '2017-07-01';  /* Start date for data processing */
SET @debug = 1;  /* Set to 1 to enable debug output */
SET @batch_size_years = 0.25;  /* Process quarterly batches for better performance */
SET @disable_keys = 0;  /* Set to 1 to disable/enable keys during update (test carefully!) */

SELECT CONCAT('Financial Update: Processing data from: ', @start_date, ', debug = ', @debug, ', batch_size_years = ', @batch_size_years) AS update_info;

/* Create optimized indexes for performance */
SELECT "Creating performance indexes..." AS index_creation;
SET @index_sql = 'CREATE INDEX IF NOT EXISTS idx_ts_code_ann_date_report ON ts_a_stock_financial_profile (ts_code, ann_date, report_period)';
PREPARE stmt FROM @index_sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @index_sql = 'CREATE INDEX IF NOT EXISTS idx_w_symbol_link ON ts_link_table (w_symbol, link_symbol)';
PREPARE stmt FROM @index_sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @index_sql = 'CREATE INDEX IF NOT EXISTS idx_ann_date_ts_code ON ts_a_stock_financial_profile (ann_date, ts_code)';
PREPARE stmt FROM @index_sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SELECT "Index creation completed." AS index_status;

/* No initialization needed - records will be created during updates */

/* Update financial metrics from ts_a_stock_financial_profile based on announcement dates */
SELECT "Update financial metrics from ts_a_stock_financial_profile based on announcement dates (batch processing by tradedate)" as info;

/* Get the last financial update time from the tracking table */
SET @financial_update_start = COALESCE(
    (SELECT MAX(end_day) FROM update_record_table
     WHERE update_type = 'financial_profile'),
    @start_date
);

SELECT CONCAT('Last financial update was on: ', @financial_update_start) AS last_update_info;

/* Get the date range that needs updating */
SET @max_tradedate = (SELECT MAX(tradedate) FROM final_a_stock_comb_info);
SET @current_batch_start = @financial_update_start;
SET @total_updated_records = 0;

SELECT CONCAT('Processing date range: ', @financial_update_start, ' to ', @max_tradedate) AS processing_range;

/* Create a temporary table to store batch processing dates */
DROP TEMPORARY TABLE IF EXISTS temp_batch_dates;
CREATE TEMPORARY TABLE temp_batch_dates (
    batch_start DATE PRIMARY KEY,
    batch_end DATE
);

/* Create stored procedure for batch processing */
DELIMITER //
DROP PROCEDURE IF EXISTS process_financial_profile_batches //
CREATE PROCEDURE process_financial_profile_batches()
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE batch_start_date DATE;
    DECLARE batch_end_date DATE;
    DECLARE cur CURSOR FOR SELECT batch_start, batch_end FROM temp_batch_dates ORDER BY batch_start;
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SELECT 'Error occurred during batch processing. Transaction rolled back.' AS error_message;
        RESIGNAL;
    END;

    /* Generate batch date ranges (quarterly batches) */
    SET @current_date = @financial_update_start;
    batch_date_loop: LOOP
        SET @batch_end = DATE_SUB(DATE_ADD(@current_date, INTERVAL @batch_size_years YEAR), INTERVAL 1 DAY);

        IF @batch_end > @max_tradedate THEN
            SET @batch_end = @max_tradedate;
        END IF;

        INSERT INTO temp_batch_dates VALUES (@current_date, @batch_end);

        IF @batch_end >= @max_tradedate THEN
            LEAVE batch_date_loop;
        END IF;

        SET @current_date = DATE_ADD(@current_date, INTERVAL @batch_size_years YEAR);
    END LOOP;

    SELECT CONCAT('Generated ', (SELECT COUNT(*) FROM temp_batch_dates), ' batches for processing') AS batch_info;

    /* Pre-compute report ranges for new announcements using window functions */
    CREATE TEMPORARY TABLE temp_report_ranges AS
    SELECT
        ts_code,
        ann_date,
        LEAD(ann_date) OVER (PARTITION BY ts_code ORDER BY ann_date) AS next_ann_date,
        report_period,
        /* All financial metrics to be updated */
        current_ratio, quick_ratio, cash_ratio,
        ca_turn, inv_turn, ar_turn, fa_turn, assets_turn,
        roic, roe_ttm, roa_ttm, grossprofit_margin_ttm, netprofit_margin_ttm, fcf_margin_ttm,
        debt_to_assets, debt_to_eqt, debt_to_ebitda,
        bps, eps_ttm, revenue_ps_ttm, cfps, fcff_ps,
        or_yoy, netprofit_yoy, basic_eps_yoy, equity_yoy, assets_yoy, ocf_yoy, roe_yoy,
        revenue_cagr_3y, netincome_cagr_3y,
        rd_exp_to_capex, goodwill
    FROM ts_a_stock_financial_profile
    WHERE ann_date > @financial_update_start AND ann_date >= @start_date;

    /* Fill NULL next_ann_date with far future date to avoid range issues */
    UPDATE temp_report_ranges SET next_ann_date = '2100-01-01' WHERE next_ann_date IS NULL;

    /* Add index on temp table for faster JOINs */
    ALTER TABLE temp_report_ranges ADD INDEX idx_ts_code_ann_date (ts_code, ann_date);

    /* Determine minimum affected tradedate (new reports start) */
    SET @min_affected_tradedate = (SELECT MIN(ann_date) FROM temp_report_ranges);
    IF @min_affected_tradedate IS NULL THEN
        SET @min_affected_tradedate = @financial_update_start;  /* No new reports, fallback */
    END IF;

    IF @debug = 1 THEN
        SELECT CONCAT('Precomputed report ranges for ', (SELECT COUNT(*) FROM temp_report_ranges), ' new announcements') AS precompute_info;
        SELECT 'Sample of precomputed ranges:' AS sample_info;
        SELECT ts_code, ann_date, next_ann_date, report_period, current_ratio
        FROM temp_report_ranges ORDER BY ts_code, ann_date LIMIT 5;
    END IF;

    /* Optionally disable keys for faster bulk updates */
    IF @disable_keys = 1 THEN
        ALTER TABLE final_a_stock_comb_info DISABLE KEYS;
        SELECT "Disabled keys on final_a_stock_comb_info for faster updates" AS key_management;
    END IF;

    OPEN cur;

    batch_loop: LOOP
        FETCH cur INTO batch_start_date, batch_end_date;
        IF done THEN
            LEAVE batch_loop;
        END IF;

        /* Skip batches that don't overlap with affected tradedates (incremental optimization) */
        IF batch_end_date < @min_affected_tradedate THEN
            IF @debug = 1 THEN
                SELECT CONCAT('Skipping batch: ', batch_start_date, ' to ', batch_end_date, ' (before affected range)') AS skip_info;
            END IF;
            ITERATE batch_loop;  /* Skip to next batch */
        END IF;

        /* Start transaction for this batch */
        START TRANSACTION;

        IF @debug = 1 THEN
            SELECT CONCAT('Processing batch: ', batch_start_date, ' to ', batch_end_date) AS batch_status;
        END IF;

        /* Bulk update financial metrics for this batch using precomputed ranges */
        UPDATE final_a_stock_comb_info target
        INNER JOIN ts_link_table link ON target.symbol = link.w_symbol
        INNER JOIN temp_report_ranges r ON r.ts_code = link.link_symbol
        SET
            /* Liquidity ratios */
            target.current_ratio = r.current_ratio,
            target.quick_ratio = r.quick_ratio,
            target.cash_ratio = r.cash_ratio,

            /* Turnover ratios */
            target.ca_turn = r.ca_turn,
            target.inv_turn = r.inv_turn,
            target.ar_turn = r.ar_turn,
            target.fa_turn = r.fa_turn,
            target.assets_turn = r.assets_turn,

            /* Profitability - Basic ratios that don't need TTM calculation */
            target.roic = r.roic,
            target.roe_ttm = r.roe_ttm,
            target.roa_ttm = r.roa_ttm,
            target.grossprofit_margin_ttm = r.grossprofit_margin_ttm,
            target.netprofit_margin_ttm = r.netprofit_margin_ttm,
            target.fcf_margin_ttm = r.fcf_margin_ttm,

            /* Leverage */
            target.debt_to_assets = r.debt_to_assets,
            target.debt_to_eqt = r.debt_to_eqt,
            target.debt_to_ebitda = r.debt_to_ebitda,

            /* Valuation */
            target.bps = r.bps,
            target.eps_ttm = r.eps_ttm,
            target.revenue_ps_ttm = r.revenue_ps_ttm,
            target.cfps = r.cfps,
            target.fcff_ps = r.fcff_ps,

            /* Growth */
            target.or_yoy = r.or_yoy,
            target.netprofit_yoy = r.netprofit_yoy,
            target.basic_eps_yoy = r.basic_eps_yoy,
            target.equity_yoy = r.equity_yoy,
            target.assets_yoy = r.assets_yoy,
            target.ocf_yoy = r.ocf_yoy,
            target.roe_yoy = r.roe_yoy,
            target.revenue_cagr_3y = r.revenue_cagr_3y,
            target.netincome_cagr_3y = r.netincome_cagr_3y,

            /* Other */
            target.rd_exp_to_capex = r.rd_exp_to_capex,
            target.goodwill = r.goodwill
        WHERE target.tradedate BETWEEN batch_start_date AND batch_end_date
          AND target.tradedate > r.ann_date  /* Apply from day after announcement */
          AND target.tradedate <= r.next_ann_date;  /* Until next announcement (inclusive) */

        SET @batch_updated = ROW_COUNT();
        SET @total_updated_records = @total_updated_records + @batch_updated;

        IF @debug = 1 THEN
            SELECT CONCAT('Batch ', batch_start_date, '-', batch_end_date, ': Updated ', @batch_updated, ' records') AS batch_result;
        END IF;

        /* Record individual batch update in tracking table */
        IF @batch_updated > 0 THEN
            INSERT INTO update_record_table (
                update_type, end_day, start_day, record_count, last_update_time
            ) VALUES (
                'financial_profile',
                batch_end_date,
                batch_start_date,
                @batch_updated,
                NOW()
            );
        END IF;

        /* Commit the batch transaction */
        COMMIT;

    END LOOP;

    CLOSE cur;

    /* Re-enable keys if they were disabled */
    IF @disable_keys = 1 THEN
        ALTER TABLE final_a_stock_comb_info ENABLE KEYS;
        SELECT "Re-enabled keys on final_a_stock_comb_info" AS key_management;
    END IF;

    /* Clean up temporary table */
    DROP TEMPORARY TABLE IF EXISTS temp_report_ranges;
END //
DELIMITER ;

/* Execute the batch processing procedure */
CALL process_financial_profile_batches();


/* Show overall update summary */
SELECT
    CONCAT('Financial metrics batch update completed. Total affected rows: ', @total_updated_records) AS update_summary,
    CONCAT('Updated records from: ', @financial_update_start, ' to ', @max_tradedate) AS update_range,
    CONCAT('Processed in ', (SELECT COUNT(*) FROM temp_batch_dates), ' batches') AS batch_summary;

/* Show batch details */
SELECT 'Batch processing details:' as info;
SELECT
    update_type,
    start_day,
    end_day,
    record_count,
    last_update_time
FROM update_record_table
WHERE update_type = 'financial_profile'
  AND last_update_time >= (SELECT MIN(last_update_time) FROM update_record_table WHERE update_type = 'financial_profile')
ORDER BY last_update_time DESC;

/* Clean up temporary table */
DROP TEMPORARY TABLE IF EXISTS temp_batch_dates;

/* Optional: Show sample of updated records for verification */
SELECT "Sample of recently updated records:" as info;
SELECT
    tradedate,
    symbol,
    current_ratio,
    roic,
    debt_to_assets,
    bps,
    or_yoy,
    netprofit_yoy,
    roe_ttm,
    eps_ttm,
    revenue_cagr_3y
FROM final_a_stock_comb_info
WHERE tradedate >= @financial_update_start
  AND (current_ratio IS NOT NULL OR roic IS NOT NULL OR roe_ttm IS NOT NULL
       OR or_yoy IS NOT NULL OR eps_ttm IS NOT NULL OR revenue_cagr_3y IS NOT NULL)
ORDER BY tradedate DESC, symbol
LIMIT 10;
