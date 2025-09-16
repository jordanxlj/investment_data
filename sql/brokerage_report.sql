/* Module 12: Update Consensus Report Data */

SELECT "Update final_a_stock_comb_info with consensus report data - use min_price as target_price" as info;
UPDATE final_a_stock_comb_info final
INNER JOIN (
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
    consensus.min_price AS f_target_price  -- Use min_price as target_price as requested
  FROM ts_a_stock_consensus_report consensus
  LEFT JOIN ts_link_table ON consensus.ts_code = ts_link_table.link_symbol
  WHERE consensus.eval_date >= @start_date
    AND consensus.eval_date > @max_tradedate
    AND (consensus.report_period LIKE '%2025%' OR consensus.report_period LIKE '2025%' OR YEAR(consensus.eval_date) >= 2025)  -- Match current year data using DATE functions
    AND consensus.total_reports > 0  -- Only include records with actual reports
) AS consensus_updates ON final.tradedate = consensus_updates.tradedate AND final.symbol = consensus_updates.symbol
SET
  final.f_pos_ratio = CASE
    WHEN consensus_updates.total_reports > 0 THEN consensus_updates.sentiment_pos / consensus_updates.total_reports
    ELSE 0
  END,
  final.f_neg_ratio = CASE
    WHEN consensus_updates.total_reports > 0 THEN consensus_updates.sentiment_neg / consensus_updates.total_reports
    ELSE 0
  END,
  final.f_eps = consensus_updates.f_eps,
  final.f_pe = consensus_updates.f_pe,
  final.f_dv_ratio = consensus_updates.f_dv_ratio,
  final.f_roe = consensus_updates.f_roe,
  final.f_target_price = consensus_updates.f_target_price;

/* Debug: Count updated rows */
SELECT ROW_COUNT() AS updated_count;
IF @debug = 1 THEN
    SELECT AVG(f_target_price) AS sample_avg FROM final_a_stock_comb_info WHERE f_target_price IS NOT NULL LIMIT 1;
END IF;