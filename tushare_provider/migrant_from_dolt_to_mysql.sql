-- 0) 目标库与前置设置
CREATE DATABASE IF NOT EXISTS investment_data_mysql DEFAULT CHARACTER SET utf8mb4;
USE investment_data_mysql;

-- 若 FEDERATED 未启用，请启用（不同平台方式略有差异）
-- INSTALL PLUGIN federated SONAME 'ha_federated.so';

-- 1) 目标表 DDL（尽量与现有结构一致）

-- 1.1 ts_index_weight（按 tushare 字段推断；如源结构不同请按源表调整）
DROP TABLE IF EXISTS ts_index_weight;
CREATE TABLE ts_index_weight (
  index_code   VARCHAR(16) NOT NULL,
  con_code     VARCHAR(16) NOT NULL,
  trade_date   VARCHAR(8)  NOT NULL,
  weight       FLOAT,
  stock_code   VARCHAR(16),
  PRIMARY KEY (index_code, con_code, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 1.2 ts_a_stock_eod_price
DROP TABLE IF EXISTS ts_a_stock_eod_price;
CREATE TABLE ts_a_stock_eod_price (
  symbol    VARCHAR(16) NOT NULL,
  tradedate DATE        NOT NULL,
  high      DECIMAL(16,4),
  low       DECIMAL(16,4),
  open      DECIMAL(16,4),
  close     DECIMAL(16,4),
  volume    BIGINT,
  adjclose  DECIMAL(16,4),
  amount    DECIMAL(16,4),
  PRIMARY KEY (symbol, tradedate),
  KEY idx_tradedate (tradedate)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 1.3 ts_a_stock_fundamental（与仓库中 daily_update.sh 定义保持一致）
DROP TABLE IF EXISTS ts_a_stock_fundamental;
CREATE TABLE ts_a_stock_fundamental (
  ts_code        VARCHAR(16) NOT NULL,
  trade_date     VARCHAR(8)  NOT NULL,
  turnover_rate  FLOAT,
  turnover_rate_f FLOAT,
  volume_ratio   FLOAT,
  pe             FLOAT,
  pe_ttm         FLOAT,
  pb             FLOAT,
  ps             FLOAT,
  ps_ttm         FLOAT,
  dv_ratio       FLOAT,
  dv_ttm         FLOAT,
  total_share    DECIMAL(16,4),
  float_share    DECIMAL(16,4),
  free_share     DECIMAL(16,4),
  total_mv       DECIMAL(16,4),
  circ_mv        DECIMAL(16,4),
  PRIMARY KEY (ts_code, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 1.4 ts_link_table（按使用场景推断）
DROP TABLE IF EXISTS ts_link_table;
CREATE TABLE ts_link_table (
  w_symbol   VARCHAR(16) NOT NULL,
  link_symbol VARCHAR(16) NOT NULL,
  link_date  DATE,
  adj_ratio  DECIMAL(16,4),
  PRIMARY KEY (link_symbol),
  KEY idx_w_symbol (w_symbol)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 1.5 final_a_stock_eod_price（与现有查询/使用保持一致）
DROP TABLE IF EXISTS final_a_stock_eod_price;
CREATE TABLE final_a_stock_eod_price (
  symbol    VARCHAR(16) NOT NULL,
  tradedate DATE        NOT NULL,
  high      DECIMAL(16,4),
  low       DECIMAL(16,4),
  open      DECIMAL(16,4),
  close     DECIMAL(16,4),
  volume    BIGINT,
  amount    DECIMAL(16,4),
  adjclose  DECIMAL(16,4),
  PRIMARY KEY (symbol, tradedate),
  KEY idx_tradedate (tradedate)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- 2) 源端（Dolt）FEDERATED 映射表
-- 说明：若不想在 CREATE TABLE 内写长 CONNECTION，可先 CREATE SERVER；这里直接用 CONNECTION 方式

DROP TABLE IF EXISTS src_ts_index_weight;
CREATE TABLE src_ts_index_weight (
  index_code   VARCHAR(16) NOT NULL,
  con_code     VARCHAR(16) NOT NULL,
  trade_date   VARCHAR(8)  NOT NULL,
  weight       FLOAT,
  stock_code   VARCHAR(16),
  PRIMARY KEY (index_code, con_code, trade_date)
) ENGINE=FEDERATED
CONNECTION='mysql://root:@127.0.0.1:3306/investment_data/ts_index_weight';

DROP TABLE IF EXISTS src_ts_a_stock_eod_price;
CREATE TABLE src_ts_a_stock_eod_price (
  symbol    VARCHAR(16) NOT NULL,
  tradedate DATE        NOT NULL,
  high      DECIMAL(16,4),
  low       DECIMAL(16,4),
  open      DECIMAL(16,4),
  close     DECIMAL(16,4),
  volume    BIGINT,
  adjclose  DECIMAL(16,4),
  amount    DECIMAL(16,4),
  PRIMARY KEY (symbol, tradedate)
) ENGINE=FEDERATED
CONNECTION='mysql://root:@127.0.0.1:3306/investment_data/ts_a_stock_eod_price';

DROP TABLE IF EXISTS src_ts_a_stock_fundamental;
CREATE TABLE src_ts_a_stock_fundamental (
  ts_code        VARCHAR(16) NOT NULL,
  trade_date     VARCHAR(8)  NOT NULL,
  turnover_rate  FLOAT,
  turnover_rate_f FLOAT,
  volume_ratio   FLOAT,
  pe             FLOAT,
  pe_ttm         FLOAT,
  pb             FLOAT,
  ps             FLOAT,
  ps_ttm         FLOAT,
  dv_ratio       FLOAT,
  dv_ttm         FLOAT,
  total_share    DECIMAL(16,4),
  float_share    DECIMAL(16,4),
  free_share     DECIMAL(16,4),
  total_mv       DECIMAL(16,4),
  circ_mv        DECIMAL(16,4),
  PRIMARY KEY (ts_code, trade_date)
) ENGINE=FEDERATED
CONNECTION='mysql://root:@127.0.0.1:3306/investment_data/ts_a_stock_fundamental';

DROP TABLE IF EXISTS src_ts_link_table;
CREATE TABLE src_ts_link_table (
  w_symbol   VARCHAR(16) NOT NULL,
  link_symbol VARCHAR(16) NOT NULL,
  link_date  DATE,
  adj_ratio  DECIMAL(16,4),
  PRIMARY KEY (link_symbol)
) ENGINE=FEDERATED
CONNECTION='mysql://root:@127.0.0.1:3306/investment_data/ts_link_table';

DROP TABLE IF EXISTS src_final_a_stock_eod_price;
CREATE TABLE src_final_a_stock_eod_price (
  symbol    VARCHAR(16) NOT NULL,
  tradedate DATE        NOT NULL,
  high      DECIMAL(16,4),
  low       DECIMAL(16,4),
  open      DECIMAL(16,4),
  close     DECIMAL(16,4),
  volume    BIGINT,
  amount    DECIMAL(16,4),
  adjclose  DECIMAL(16,4),
  PRIMARY KEY (symbol, tradedate)
) ENGINE=FEDERATED
CONNECTION='mysql://root:@127.0.0.1:3306/investment_data/final_a_stock_eod_price';


-- 3) 数据拷贝（可分表执行；大表建议分批带 WHERE 条件按日期段搬迁）
SET sql_safe_updates=0;

INSERT INTO ts_index_weight
SELECT * FROM src_ts_index_weight;

INSERT INTO ts_a_stock_eod_price
SELECT * FROM src_ts_a_stock_eod_price;

INSERT INTO ts_a_stock_fundamental
SELECT * FROM src_ts_a_stock_fundamental;

INSERT INTO ts_link_table
SELECT * FROM src_ts_link_table;

INSERT INTO final_a_stock_eod_price
SELECT * FROM src_final_a_stock_eod_price;


-- 4) 校验（行数/范围/聚合/抽样差异）

-- 4.1 行数
SELECT 'ts_index_weight' AS tbl,
       (SELECT COUNT(*) FROM src_ts_index_weight) AS src_cnt,
       (SELECT COUNT(*) FROM ts_index_weight)     AS dst_cnt
UNION ALL
SELECT 'ts_a_stock_eod_price',
       (SELECT COUNT(*) FROM src_ts_a_stock_eod_price),
       (SELECT COUNT(*) FROM ts_a_stock_eod_price)
UNION ALL
SELECT 'ts_a_stock_fundamental',
       (SELECT COUNT(*) FROM src_ts_a_stock_fundamental),
       (SELECT COUNT(*) FROM ts_a_stock_fundamental)
UNION ALL
SELECT 'ts_link_table',
       (SELECT COUNT(*) FROM src_ts_link_table),
       (SELECT COUNT(*) FROM ts_link_table)
UNION ALL
SELECT 'final_a_stock_eod_price',
       (SELECT COUNT(*) FROM src_final_a_stock_eod_price),
       (SELECT COUNT(*) FROM final_a_stock_eod_price);

-- 4.2 日期范围
SELECT 'ts_a_stock_eod_price' AS tbl,
       (SELECT MIN(tradedate) FROM src_ts_a_stock_eod_price) AS src_min,
       (SELECT MAX(tradedate) FROM src_ts_a_stock_eod_price) AS src_max,
       (SELECT MIN(tradedate) FROM ts_a_stock_eod_price)     AS dst_min,
       (SELECT MAX(tradedate) FROM ts_a_stock_eod_price)     AS dst_max
UNION ALL
SELECT 'final_a_stock_eod_price',
       (SELECT MIN(tradedate) FROM src_final_a_stock_eod_price),
       (SELECT MAX(tradedate) FROM src_final_a_stock_eod_price),
       (SELECT MIN(tradedate) FROM final_a_stock_eod_price),
       (SELECT MAX(tradedate) FROM final_a_stock_eod_price);

SELECT 'ts_a_stock_fundamental' AS tbl,
       (SELECT MIN(trade_date) FROM src_ts_a_stock_fundamental) AS src_min,
       (SELECT MAX(trade_date) FROM src_ts_a_stock_fundamental) AS src_max,
       (SELECT MIN(trade_date) FROM ts_a_stock_fundamental)     AS dst_min,
       (SELECT MAX(trade_date) FROM ts_a_stock_fundamental)     AS dst_max;

-- 4.3 数值聚合（大偏差预警；浮点有微小误差属正常）
SELECT 'ts_a_stock_eod_price' AS tbl,
       (SELECT SUM(volume) FROM src_ts_a_stock_eod_price) AS src_sum_vol,
       (SELECT SUM(amount) FROM src_ts_a_stock_eod_price) AS src_sum_amt,
       (SELECT SUM(volume) FROM ts_a_stock_eod_price)     AS dst_sum_vol,
       (SELECT SUM(amount) FROM ts_a_stock_eod_price)     AS dst_sum_amt
UNION ALL
SELECT 'final_a_stock_eod_price',
       (SELECT SUM(volume) FROM src_final_a_stock_eod_price),
       (SELECT SUM(amount) FROM src_final_a_stock_eod_price),
       (SELECT SUM(volume) FROM final_a_stock_eod_price),
       (SELECT SUM(amount) FROM final_a_stock_eod_price);

SELECT 'ts_a_stock_fundamental' AS tbl,
       (SELECT SUM(total_share) FROM src_ts_a_stock_fundamental) AS src_sum_total_share,
       (SELECT SUM(circ_mv)     FROM src_ts_a_stock_fundamental) AS src_sum_circ_mv,
       (SELECT SUM(total_share) FROM ts_a_stock_fundamental)     AS dst_sum_total_share,
       (SELECT SUM(circ_mv)     FROM ts_a_stock_fundamental)     AS dst_sum_circ_mv;

-- 4.4 抽样差异（EOD 样例）
SELECT 'ts_a_stock_eod_price' AS tbl, COUNT(*) AS only_in_src
FROM src_ts_a_stock_eod_price s
LEFT JOIN ts_a_stock_eod_price d ON d.symbol=s.symbol AND d.tradedate=s.tradedate
WHERE d.symbol IS NULL
UNION ALL
SELECT 'ts_a_stock_eod_price', COUNT(*) AS only_in_dst
FROM ts_a_stock_eod_price d
LEFT JOIN src_ts_a_stock_eod_price s ON d.symbol=s.symbol AND d.tradedate=s.tradedate
WHERE s.symbol IS NULL;
