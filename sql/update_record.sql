/* Create update_record_table if it doesn't exist */
CREATE TABLE IF NOT EXISTS update_record_table (
    id INT AUTO_INCREMENT PRIMARY KEY,
    update_type VARCHAR(64) NOT NULL,
    start_day DATE NOT NULL,
    end_day DATE NOT NULL,
    last_update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    record_count INT DEFAULT 0,
    INDEX idx_update_type_end_day (update_type, end_day)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Tracks detailed update batches for all data tables';
