##必需的索引（按优先级排序）

### ts_link_table 表（最重要！）：
    - INDEX idx_link_symbol (link_symbol) - 用于 JOIN ts_code
    - INDEX idx_w_symbol (w_symbol) - 用于 JOIN symbol
    - UNIQUE INDEX uk_w_symbol_link_symbol (w_symbol, link_symbol)

### ts_a_stock_eod_price 表：
    - INDEX idx_symbol_tradedate (symbol, tradedate) - 用于 JOIN 和 WHERE
    - INDEX idx_tradedate (tradedate) - 用于日期范围查询

### ts_a_stock_fundamental 表：
    - INDEX idx_ts_code_trade_date (ts_code, trade_date) - 用于 JOIN 和 WHERE
    - INDEX idx_trade_date (trade_date) - 用于日期过滤

### ts_a_stock_moneyflow 表：
    - INDEX idx_ts_code_trade_date (ts_code, trade_date) - 用于 JOIN 和 WHERE

### ts_a_stock_cost_pct 表：
    - INDEX idx_ts_code_trade_date (ts_code, trade_date) - 用于 JOIN 和 WHERE

### ts_a_stock_suspend_info 表：
    - INDEX idx_ts_code_trade_date (ts_code, trade_date) - 用于 JOIN 和 WHERE

### ts_a_stock_consensus_report 表：
    - INDEX idx_ts_code_eval_date (ts_code, eval_date) - 用于 JOIN 和 WHERE
    - INDEX idx_eval_date (eval_date) - 用于日期过滤
