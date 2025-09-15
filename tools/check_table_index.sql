-- 检查 ts_a_stock_consensus_report 表是否存在以及其结构
SELECT 'Checking ts_a_stock_consensus_report table...' AS status;

-- ============================================================================
-- CRITICAL INDEXES CREATION SCRIPT
-- ============================================================================
-- Run this script to create all required indexes for optimal performance
-- Supports 8 tables with 16+ indexes total
-- ============================================================================

-- 1. ts_link_table 索引（最重要！）
SELECT 'Creating ts_link_table indexes...' AS status;

-- 检查是否已存在索引，如果不存在则创建
SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'ts_link_table'
       AND INDEX_NAME = 'idx_link_symbol') = 0,
    'CREATE INDEX idx_link_symbol ON ts_link_table (link_symbol)',
    'SELECT "idx_link_symbol already exists" AS status'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'ts_link_table'
       AND INDEX_NAME = 'idx_w_symbol') = 0,
    'CREATE INDEX idx_w_symbol ON ts_link_table (w_symbol)',
    'SELECT "idx_w_symbol already exists" AS status'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'ts_link_table'
       AND INDEX_NAME = 'uk_w_symbol_link_symbol') = 0,
    'CREATE UNIQUE INDEX uk_w_symbol_link_symbol ON ts_link_table (w_symbol, link_symbol)',
    'SELECT "uk_w_symbol_link_symbol already exists" AS status'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 2. ts_a_stock_eod_price 索引
SELECT 'Creating ts_a_stock_eod_price indexes...' AS status;

SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'ts_a_stock_eod_price'
       AND INDEX_NAME = 'idx_symbol_tradedate') = 0,
    'CREATE INDEX idx_symbol_tradedate ON ts_a_stock_eod_price (symbol, tradedate)',
    'SELECT "idx_symbol_tradedate already exists" AS status'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'ts_a_stock_eod_price'
       AND INDEX_NAME = 'idx_tradedate') = 0,
    'CREATE INDEX idx_tradedate ON ts_a_stock_eod_price (tradedate)',
    'SELECT "idx_tradedate already exists" AS status'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 3. ts_a_stock_fundamental 索引
SELECT 'Creating ts_a_stock_fundamental indexes...' AS status;

SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'ts_a_stock_fundamental'
       AND INDEX_NAME = 'idx_ts_code_trade_date') = 0,
    'CREATE INDEX idx_ts_code_trade_date ON ts_a_stock_fundamental (ts_code, trade_date)',
    'SELECT "idx_ts_code_trade_date already exists" AS status'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'ts_a_stock_fundamental'
       AND INDEX_NAME = 'idx_trade_date') = 0,
    'CREATE INDEX idx_trade_date ON ts_a_stock_fundamental (trade_date)',
    'SELECT "idx_trade_date already exists" AS status'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 4. ts_a_stock_moneyflow 索引
SELECT 'Creating ts_a_stock_moneyflow indexes...' AS status;

SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'ts_a_stock_moneyflow'
       AND INDEX_NAME = 'idx_ts_code_trade_date') = 0,
    'CREATE INDEX idx_ts_code_trade_date ON ts_a_stock_moneyflow (ts_code, trade_date)',
    'SELECT "idx_ts_code_trade_date already exists" AS status'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 5. ts_a_stock_cost_pct 索引
SELECT 'Creating ts_a_stock_cost_pct indexes...' AS status;

SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'ts_a_stock_cost_pct'
       AND INDEX_NAME = 'idx_ts_code_trade_date') = 0,
    'CREATE INDEX idx_ts_code_trade_date ON ts_a_stock_cost_pct (ts_code, trade_date)',
    'SELECT "idx_ts_code_trade_date already exists" AS status'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 6. ts_a_stock_suspend_info 索引
SELECT 'Creating ts_a_stock_suspend_info indexes...' AS status;

SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'ts_a_stock_suspend_info'
       AND INDEX_NAME = 'idx_ts_code_trade_date') = 0,
    'CREATE INDEX idx_ts_code_trade_date ON ts_a_stock_suspend_info (ts_code, trade_date)',
    'SELECT "idx_ts_code_trade_date already exists" AS status'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 7. ts_a_stock_consensus_report 索引
SELECT 'Creating ts_a_stock_consensus_report indexes...' AS status;

SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'ts_a_stock_consensus_report'
       AND INDEX_NAME = 'idx_ts_code_eval_date') = 0,
    'CREATE INDEX idx_ts_code_eval_date ON ts_a_stock_consensus_report (ts_code, eval_date)',
    'SELECT "idx_ts_code_eval_date already exists" AS status'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'ts_a_stock_consensus_report'
       AND INDEX_NAME = 'idx_eval_date') = 0,
    'CREATE INDEX idx_eval_date ON ts_a_stock_consensus_report (eval_date)',
    'SELECT "idx_eval_date already exists" AS status'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 8. ts_a_stock_financial_profile 索引（用于 TTM/CAGR 计算）
SELECT 'Creating ts_a_stock_financial_profile indexes...' AS status;

-- 最重要：ts_code + ann_date 复合索引
SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'ts_a_stock_financial_profile'
       AND INDEX_NAME = 'idx_ts_code_ann_date') = 0,
    'CREATE INDEX idx_ts_code_ann_date ON ts_a_stock_financial_profile (ts_code, ann_date)',
    'SELECT "idx_ts_code_ann_date already exists" AS status'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 完全覆盖WHERE条件的复合索引
SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'ts_a_stock_financial_profile'
       AND INDEX_NAME = 'idx_ts_code_report_period_ann_date') = 0,
    'CREATE INDEX idx_ts_code_report_period_ann_date ON ts_a_stock_financial_profile (ts_code, report_period, ann_date)',
    'SELECT "idx_ts_code_report_period_ann_date already exists" AS status'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 日期范围查询索引
SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'ts_a_stock_financial_profile'
       AND INDEX_NAME = 'idx_ann_date') = 0,
    'CREATE INDEX idx_ann_date ON ts_a_stock_financial_profile (ann_date)',
    'SELECT "idx_ann_date already exists" AS status'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 报告期过滤索引
SET @sql = (SELECT IF(
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
     WHERE TABLE_SCHEMA = DATABASE()
       AND TABLE_NAME = 'ts_a_stock_financial_profile'
       AND INDEX_NAME = 'idx_report_period') = 0,
    'CREATE INDEX idx_report_period ON ts_a_stock_financial_profile (report_period)',
    'SELECT "idx_report_period already exists" AS status'
));
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SELECT 'All required indexes have been created or already exist!' AS final_status;
SELECT 'Total indexes created: 16 (8 tables × average 2 indexes per table)' AS summary;

-- ============================================================================
-- INDEX ANALYSIS AND OPTIMIZATION
-- ============================================================================
-- Analyze existing indexes and identify potentially unnecessary ones
-- ============================================================================

-- 1. 检查所有表的当前索引状态
SELECT '=== Current Index Status Analysis ===' AS section;

SELECT
    TABLE_NAME,
    INDEX_NAME,
    COLUMN_NAME,
    SEQ_IN_INDEX,
    CARDINALITY,
    INDEX_TYPE,
    NON_UNIQUE
FROM INFORMATION_SCHEMA.STATISTICS
WHERE TABLE_SCHEMA = DATABASE()
ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX;

-- 2. 分析复合索引 vs 单列索引的冗余性
SELECT '=== Composite Index Analysis ===' AS section;

-- 检查是否有冗余的单列索引（当复合索引存在时）
SELECT
    t.TABLE_NAME,
    t.INDEX_NAME as COMPOSITE_INDEX,
    GROUP_CONCAT(t.COLUMN_NAME ORDER BY t.SEQ_IN_INDEX) as COMPOSITE_COLUMNS,
    s.INDEX_NAME as SINGLE_INDEX,
    s.COLUMN_NAME as SINGLE_COLUMN
FROM INFORMATION_SCHEMA.STATISTICS t
JOIN INFORMATION_SCHEMA.STATISTICS s ON t.TABLE_SCHEMA = s.TABLE_SCHEMA
    AND t.TABLE_NAME = s.TABLE_NAME
    AND t.INDEX_NAME != s.INDEX_NAME
    AND t.COLUMN_NAME = s.COLUMN_NAME
    AND t.SEQ_IN_INDEX = 1
    AND s.SEQ_IN_INDEX = 1
WHERE t.TABLE_SCHEMA = DATABASE()
    AND t.NON_UNIQUE = 1
    AND s.NON_UNIQUE = 1
    AND (
        SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
        WHERE TABLE_SCHEMA = t.TABLE_SCHEMA
          AND TABLE_NAME = t.TABLE_NAME
          AND INDEX_NAME = t.INDEX_NAME
    ) > 1  -- 复合索引
ORDER BY t.TABLE_NAME, t.INDEX_NAME;

-- 3. 检查低基数列的索引（基数 < 10 的列索引可能不必要）
SELECT '=== Low Cardinality Index Analysis ===' AS section;

SELECT
    TABLE_NAME,
    COLUMN_NAME,
    CARDINALITY,
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS c
     WHERE c.TABLE_SCHEMA = s.TABLE_SCHEMA
       AND c.TABLE_NAME = s.TABLE_NAME
       AND c.COLUMN_NAME = s.COLUMN_NAME) as TOTAL_ROWS,
    ROUND(CARDINALITY / (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS c
                         WHERE c.TABLE_SCHEMA = s.TABLE_SCHEMA
                           AND c.TABLE_NAME = s.TABLE_NAME
                           AND c.COLUMN_NAME = s.COLUMN_NAME) * 100, 2) as SELECTIVITY_PERCENT
FROM INFORMATION_SCHEMA.STATISTICS s
WHERE TABLE_SCHEMA = DATABASE()
    AND CARDINALITY < 10
    AND NON_UNIQUE = 1
ORDER BY CARDINALITY;

-- 4. 检查从未使用过的索引（需要 SHOW INDEX STATISTICS 或 performance_schema）
SELECT '=== Index Usage Analysis (if available) ===' AS section;

-- 检查是否有 performance_schema.index_statistics 表
SELECT TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'performance_schema'
  AND TABLE_NAME = 'index_statistics'
LIMIT 1;

-- 如果有 performance_schema，显示索引使用统计
SELECT 'Index usage statistics from performance_schema:' AS info;
SELECT
    OBJECT_SCHEMA,
    OBJECT_NAME,
    INDEX_NAME,
    COUNT_READ,
    COUNT_WRITE,
    COUNT_FETCH
FROM performance_schema.index_statistics
WHERE OBJECT_SCHEMA = DATABASE()
ORDER BY COUNT_READ DESC, COUNT_WRITE DESC;

-- 5. 建议删除的索引
SELECT '=== RECOMMENDED INDEXES TO DROP ===' AS section;

-- 基于分析结果的建议
SELECT 'Potential redundant indexes to consider dropping:' AS recommendation;
SELECT
    TABLE_NAME,
    INDEX_NAME,
    GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) as COLUMNS,
    CARDINALITY,
    CASE
        WHEN CARDINALITY < 5 THEN 'VERY LOW CARDINALITY - CONSIDER DROPPING'
        WHEN CARDINALITY < 10 THEN 'LOW CARDINALITY - REVIEW USAGE'
        WHEN CARDINALITY < 100 THEN 'MODERATE CARDINALITY - KEEP IF FREQUENTLY USED'
        ELSE 'GOOD CARDINALITY - KEEP'
    END as RECOMMENDATION
FROM (
    SELECT
        TABLE_NAME,
        INDEX_NAME,
        COLUMN_NAME,
        SEQ_IN_INDEX,
        CARDINALITY,
        COUNT(*) OVER (PARTITION BY TABLE_NAME, INDEX_NAME) as INDEX_COL_COUNT
    FROM INFORMATION_SCHEMA.STATISTICS
    WHERE TABLE_SCHEMA = DATABASE()
) t
WHERE SEQ_IN_INDEX = 1
ORDER BY CARDINALITY, TABLE_NAME;

-- 6. 索引大小分析
SELECT '=== Index Size Analysis ===' AS section;

SELECT
    TABLE_NAME,
    INDEX_NAME,
    ROUND(SUM(DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024, 2) as SIZE_MB,
    ROUND(AVG(CARDINALITY), 0) as AVG_CARDINALITY
FROM INFORMATION_SCHEMA.STATISTICS s
JOIN INFORMATION_SCHEMA.TABLES t ON s.TABLE_SCHEMA = t.TABLE_SCHEMA
    AND s.TABLE_NAME = t.TABLE_NAME
WHERE s.TABLE_SCHEMA = DATABASE()
GROUP BY TABLE_NAME, INDEX_NAME
ORDER BY SIZE_MB DESC;

SELECT '=== Index Analysis Complete ===' AS final_analysis;

-- ============================================================================
-- SAFE INDEX DROPPING SCRIPT
-- ============================================================================
-- Only drop indexes after careful analysis and monitoring
-- ============================================================================

SELECT '=== SAFE INDEX DROPPING RECOMMENDATIONS ===' AS section;

-- 1. 检查哪些索引可以安全删除
SELECT 'Potentially safe to drop (after monitoring):' AS info;

-- 低选择性索引（选择性 < 5%）
SELECT
    TABLE_NAME,
    INDEX_NAME,
    COLUMN_NAME,
    CARDINALITY,
    ROUND(CARDINALITY / TABLE_ROWS * 100, 2) as SELECTIVITY_PERCENT,
    'LOW SELECTIVITY - CANDIDATE FOR DROPPING' as RECOMMENDATION
FROM INFORMATION_SCHEMA.STATISTICS s
JOIN INFORMATION_SCHEMA.TABLES t ON s.TABLE_SCHEMA = t.TABLE_SCHEMA
    AND s.TABLE_NAME = t.TABLE_NAME
WHERE s.TABLE_SCHEMA = DATABASE()
    AND s.NON_UNIQUE = 1
    AND CARDINALITY / TABLE_ROWS * 100 < 5
    AND TABLE_ROWS > 1000  -- 只考虑有足够数据的表
ORDER BY SELECTIVITY_PERCENT;

-- 2. 生成删除索引的SQL（注释掉的，需要手动启用）
SELECT '=== INDEX DROP SQL (CAUTION - REVIEW BEFORE EXECUTING) ===' AS section;

/*
-- 示例删除语句（请先备份数据库！）
-- DROP INDEX idx_tradedate ON ts_a_stock_eod_price;
-- DROP INDEX idx_symbol ON final_a_stock_comb_info;
-- DROP INDEX idx_tradedate ON final_a_stock_comb_info;
*/

SELECT 'Generated DROP INDEX statements (review carefully):' AS warning;

-- 生成安全的删除语句
SELECT
    CONCAT('-- DROP INDEX ', INDEX_NAME, ' ON ', TABLE_NAME, ';') as DROP_STATEMENT,
    CONCAT('Reason: ', CASE
        WHEN CARDINALITY < 10 THEN 'Very low cardinality'
        WHEN TABLE_NAME = 'final_a_stock_comb_info' AND INDEX_NAME LIKE 'idx_%' THEN 'Covered by PRIMARY KEY'
        ELSE 'Review usage patterns'
    END) as REASON
FROM INFORMATION_SCHEMA.STATISTICS
WHERE TABLE_SCHEMA = DATABASE()
    AND NON_UNIQUE = 1
    AND (
        CARDINALITY < 10  -- 低基数
        OR (TABLE_NAME = 'final_a_stock_comb_info' AND INDEX_NAME = 'idx_symbol')  -- 主键已覆盖
        OR (TABLE_NAME = 'ts_a_stock_eod_price' AND INDEX_NAME = 'idx_tradedate')  -- 复合索引已覆盖
        OR (TABLE_NAME = 'ts_a_stock_financial_profile' AND INDEX_NAME = 'idx_ann_date' AND
            EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.STATISTICS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = 'ts_a_stock_financial_profile'
                      AND INDEX_NAME = 'idx_ts_code_ann_date'))  -- 复合索引已覆盖单列索引
    )
ORDER BY TABLE_NAME, INDEX_NAME;

-- 3. 监控建议
SELECT '=== MONITORING RECOMMENDATIONS ===' AS section;

SELECT 'Before dropping any indexes:' AS step1;
SELECT '1. Enable performance_schema if not already enabled' AS action1;
SELECT '2. Monitor index usage for 1-2 weeks' AS action2;
SELECT '3. Check slow query log for affected queries' AS action3;
SELECT '4. Test in staging environment first' AS action4;

SELECT 'After dropping indexes:' AS step2;
SELECT '1. Monitor query performance' AS action5;
SELECT '2. Check for increased CPU/memory usage' AS action6;
SELECT '3. Monitor application response times' AS action7;
SELECT '4. Be prepared to recreate indexes if needed' AS action8;

SELECT '=== END OF ANALYSIS ===' AS end_message;

-- 调试 INSERT 查询失败的原因
SELECT '=== Debugging INSERT query ===' AS section;

-- 检查 ts_link_table 的结构和数据
DESCRIBE ts_link_table;

SELECT 'Sample ts_link_table data:' AS info;
SELECT * FROM ts_link_table LIMIT 5;

SELECT CONCAT('Total records in ts_link_table: ', COUNT(*)) AS count_info FROM ts_link_table;

-- 检查 adj_ratio 的分布
SELECT 'adj_ratio distribution:' AS info;
SELECT adj_ratio, COUNT(*) as count
FROM ts_link_table
GROUP BY adj_ratio
ORDER BY count DESC;

-- 检查有多少股票的 adj_ratio 是 NULL
SELECT CONCAT('Stocks with NULL adj_ratio: ', COUNT(*)) AS null_adj_ratio_count
FROM ts_link_table
WHERE adj_ratio IS NULL;

-- 检查 ts_a_stock_eod_price 表
SELECT '=== Checking ts_a_stock_eod_price ===' AS section;
DESCRIBE ts_a_stock_eod_price;

SELECT CONCAT('Total records in ts_a_stock_eod_price: ', COUNT(*)) AS eod_count FROM ts_a_stock_eod_price;

SELECT 'Sample EOD data:' AS info;
SELECT * FROM ts_a_stock_eod_price LIMIT 3;

-- 检查日期范围
SELECT 'EOD data date range:' AS info;
SELECT MIN(tradedate) as min_date, MAX(tradedate) as max_date FROM ts_a_stock_eod_price;

-- 检查 @start_date 变量的值
SELECT '@start_date variable check:' AS info;
SET @start_date = '2025-09-01';
SELECT @start_date as current_start_date;

-- 检查符合条件的记录数量
SELECT '=== Checking matching records ===' AS section;
SELECT CONCAT('Records matching adj_ratio IS NULL: ', COUNT(*)) AS matching_stocks
FROM ts_link_table
WHERE adj_ratio IS NULL;

-- 检查在指定日期范围内的 EOD 记录
SELECT CONCAT('EOD records >= @start_date: ', COUNT(*)) AS recent_eod_records
FROM ts_a_stock_eod_price
WHERE tradedate >= @start_date;

-- 检查实际的匹配查询结果
SELECT '=== Actual matching query result ===' AS section;
SELECT COUNT(*) as total_matches
FROM ts_a_stock_eod_price eod,
     (SELECT DISTINCT link_symbol as w_missing_symbol, w_symbol
      FROM ts_link_table
      WHERE adj_ratio IS NULL) missing_table
WHERE eod.symbol = missing_table.w_missing_symbol
  AND eod.tradedate >= @start_date;

-- 显示一些匹配的样例（如果有的话）
SELECT 'Sample matches (if any):' AS info;
SELECT eod.tradedate, eod.symbol, missing_table.w_symbol, eod.close
FROM ts_a_stock_eod_price eod,
     (SELECT DISTINCT link_symbol as w_missing_symbol, w_symbol
      FROM ts_link_table
      WHERE adj_ratio IS NULL) missing_table
WHERE eod.symbol = missing_table.w_missing_symbol
  AND eod.tradedate >= @start_date
LIMIT 5;
