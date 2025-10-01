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
SET @start_date = '2010-01-01';  /* Start date for data processing */
SET @debug = 0;  /* Set to 1 to enable debug output */
SET @batch_size_years = 1;  /* Process one year at a time */

SELECT CONCAT('Financial Update: Processing data from: ', @start_date, ', debug = ', @debug, ', batch_size_years = ', @batch_size_years) AS update_info;

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

/* Generate batch date ranges (one batch per year) */
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

/* Process each batch */
DELIMITER //
batch_processing: BEGIN
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

    OPEN cur;

    batch_loop: LOOP
        FETCH cur INTO batch_start_date, batch_end_date;
        IF done THEN
            LEAVE batch_loop;
        END IF;

        /* Start transaction for this batch */
        START TRANSACTION;

        IF @debug = 1 THEN
            SELECT CONCAT('Processing batch: ', batch_start_date, ' to ', batch_end_date) AS batch_status;
        END IF;

        /* Update financial metrics for this batch */
        UPDATE final_a_stock_comb_info target
        INNER JOIN ts_link_table link ON target.symbol = link.w_symbol
        INNER JOIN ts_a_stock_financial_profile financial ON (
            financial.ts_code = link.link_symbol
            AND target.tradedate BETWEEN batch_start_date AND batch_end_date
            AND target.tradedate > financial.ann_date
            AND financial.ann_date > @financial_update_start
            AND financial.ann_date >= @start_date
            AND financial.report_period = (
                SELECT MAX(f2.report_period)
                FROM ts_a_stock_financial_profile f2
                WHERE f2.ts_code = financial.ts_code
                  AND f2.ann_date < target.tradedate
            )
        )
        SET
            /* Liquidity ratios */
            target.current_ratio = financial.current_ratio,
            target.quick_ratio = financial.quick_ratio,
            target.cash_ratio = financial.cash_ratio,

            /* Turnover ratios */
            target.ca_turn = financial.ca_turn,
            target.inv_turn = financial.inv_turn,
            target.ar_turn = financial.ar_turn,
            target.fa_turn = financial.fa_turn,
            target.assets_turn = financial.assets_turn,

            /* Profitability - Basic ratios that don't need TTM calculation */
            target.roic = financial.roic,
            target.roe_ttm = financial.roe_ttm,
            target.roa_ttm = financial.roa_ttm,
            target.grossprofit_margin_ttm = financial.grossprofit_margin_ttm,
            target.netprofit_margin_ttm = financial.netprofit_margin_ttm,
            target.fcf_margin_ttm = financial.fcf_margin_ttm,

            /* Leverage */
            target.debt_to_assets = financial.debt_to_assets,
            target.debt_to_eqt = financial.debt_to_eqt,
            target.debt_to_ebitda = financial.debt_to_ebitda,

            /* Valuation */
            target.bps = financial.bps,
            target.eps_ttm = financial.eps_ttm,
            target.revenue_ps_ttm = financial.revenue_ps_ttm,
            target.cfps = financial.cfps,
            target.fcff_ps = financial.fcff_ps,

            /* Growth */
            target.or_yoy = financial.or_yoy,
            target.netprofit_yoy = financial.netprofit_yoy,
            target.basic_eps_yoy = financial.basic_eps_yoy,
            target.equity_yoy = financial.equity_yoy,
            target.assets_yoy = financial.assets_yoy,
            target.ocf_yoy = financial.ocf_yoy,
            target.roe_yoy = financial.roe_yoy,
            target.revenue_cagr_3y = financial.revenue_cagr_3y,
            target.netincome_cagr_3y = financial.netincome_cagr_3y,

            /* Other */
            target.rd_exp_to_capex = financial.rd_exp_to_capex,
            target.goodwill = financial.goodwill;

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
END //
DELIMITER ;


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
