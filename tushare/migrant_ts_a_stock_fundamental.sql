-- 1) 新表（保留万单位，用 DECIMAL 保留小数）
CREATE TABLE IF NOT EXISTS ts_a_stock_fundamental_v2 (
    ts_code VARCHAR(16) NOT NULL,
    trade_date VARCHAR(8) NOT NULL,
    turnover_rate FLOAT,
    turnover_rate_f FLOAT,
    volume_ratio FLOAT,
    pe FLOAT,
    pe_ttm FLOAT,
    pb FLOAT,
    ps FLOAT,
    ps_ttm FLOAT,
    dv_ratio FLOAT,
    dv_ttm FLOAT,
    total_share DECIMAL(16,4),
    float_share DECIMAL(16,4),
    free_share DECIMAL(16,4),
    total_mv DECIMAL(16,4),
    circ_mv DECIMAL(16,4),
    PRIMARY KEY (ts_code, trade_date)
    );

-- 索引（如需要）
ALTER TABLE ts_a_stock_fundamental_v2
    ADD INDEX idx_trade_date (trade_date),
  ADD INDEX idx_ts_code (ts_code);

-- 2) 迁移数据（显式 CAST，避免隐式截断）
INSERT INTO ts_a_stock_fundamental_v2
(ts_code, trade_date, turnover_rate, turnover_rate_f, volume_ratio,
 pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm,
 total_share, float_share, free_share, total_mv, circ_mv)
SELECT
    ts_code, trade_date, turnover_rate, turnover_rate_f, volume_ratio,
    pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm,
    CAST(total_share AS DECIMAL(16,4)),
    CAST(float_share AS DECIMAL(16,4)),
    CAST(free_share AS DECIMAL(16,4)),
    CAST(total_mv AS DECIMAL(16,4)),
    CAST(circ_mv AS DECIMAL(16,4))
FROM ts_a_stock_fundamental;

-- 3) 原子换名
RENAME TABLE
  ts_a_stock_fundamental TO ts_a_stock_fundamental_old,
  ts_a_stock_fundamental_v2 TO ts_a_stock_fundamental;

-- 验证后再删除旧表
-- DROP TABLE ts_a_stock_fundamental_old;