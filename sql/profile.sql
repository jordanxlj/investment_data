/* ============================================================================
   Financial Metrics Update Script for final_a_stock_comb_info table

   This script updates financial metrics from ts_a_stock_financial_profile
   to final_a_stock_comb_info based on announcement dates.

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

   ⚠️  IMPORTANT: TTM and CAGR values are approximations using latest available data.
   For accurate financial analysis, consider using the Python TTM calculation scripts.

   Run this after updating ts_a_stock_financial_profile data.

   ============================================================================ */

/* Set shared variables */
SET @start_date = '2010-01-01';  /* Start date for data processing */
SET @debug = 0;  /* Set to 1 to enable debug output */

SELECT CONCAT('Financial Update: Processing data from: ', @start_date, ', debug = ', @debug) AS update_info;

/* No initialization needed - records will be created during updates */

/* Update financial metrics from ts_a_stock_financial_profile based on announcement dates */
SELECT "Update financial metrics from ts_a_stock_financial_profile based on announcement dates" as info;

/* Get the last financial update time from the tracking table */
SET @financial_update_start = COALESCE(
    (SELECT MAX(end_day) FROM update_record_table
     WHERE update_type = 'financial_profile'),
    @start_date
);

SELECT CONCAT('Last financial update was on: ', @financial_update_start) AS last_update_info;

/* Update financial metrics for existing trading records */
/* Only update records where tradedate >= ann_date (financial data available) */
/* and ann_date > last update date (new financial announcements) */
UPDATE final_a_stock_comb_info target
INNER JOIN ts_link_table link ON target.symbol = link.w_symbol
INNER JOIN ts_a_stock_financial_profile financial ON (
    financial.ts_code = link.link_symbol
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
    target.netincome_cagr_3y = financial.netprofit_cagr_3y,

    /* Other */
    target.rd_exp_to_capex = financial.rd_exp_to_capex,
    target.goodwill = financial.goodwill;


/* Record the update in the tracking table */
SET @updated_records = ROW_COUNT();
SET @max_updated_date = (SELECT MAX(tradedate) FROM final_a_stock_comb_info
                         WHERE tradedate >= @start_date
                         AND (current_ratio IS NOT NULL OR roic IS NOT NULL OR roe_ttm IS NOT NULL));

INSERT INTO update_record_table (
    update_type, end_day, start_day, record_count, last_update_time
) VALUES (
    'financial_profile',
    @max_updated_date,
    @financial_update_start,
    @updated_records,
    NOW()
);

/* Show update summary */
SELECT
    CONCAT('Financial metrics updated. Affected rows: ', ROW_COUNT()) AS update_summary,
    CONCAT('Updated records from: ', @financial_update_start, ' to present') AS update_range;

/* Show tracking table update */
SELECT 'Update tracking table:' as info, * FROM update_record_table WHERE table_name = 'final_a_stock_comb_info_financial';

/* Optional: Show sample of updated records for verification */
SELECT "Sample of updated records:" as info;
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
WHERE tradedate >= @start_date
  AND (current_ratio IS NOT NULL OR roic IS NOT NULL OR roe_ttm IS NOT NULL
       OR or_yoy IS NOT NULL OR eps_ttm IS NOT NULL OR revenue_cagr_3y IS NOT NULL)
ORDER BY tradedate DESC, symbol
LIMIT 10;
