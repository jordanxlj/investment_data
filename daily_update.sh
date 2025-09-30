#!/usr/bin/env bash
set -euo pipefail
set -x

# Config via environment variables (override as needed)
# MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB
# WORKERS, START_DATE_OVERRIDE, MIN_SYMBOLS_PER_DAY, WHERE_TRADEDATE_AFTER, CHUNKSIZE

MYSQL_HOST=${MYSQL_HOST:-127.0.0.1}
MYSQL_PORT=${MYSQL_PORT:-3307}
MYSQL_USER=${MYSQL_USER:-root}
MYSQL_PASSWORD=${MYSQL_PASSWORD:-}
MYSQL_DB=${MYSQL_DB:-investment_data_new}

WORKERS=${WORKERS:-8}
START_DATE_OVERRIDE=${START_DATE_OVERRIDE:-}
MIN_SYMBOLS_PER_DAY=${MIN_SYMBOLS_PER_DAY:-1000}
WHERE_TRADEDATE_AFTER=${WHERE_TRADEDATE_AFTER:-2023-05-01}
CHUNKSIZE=${CHUNKSIZE:-2000}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/src"

MYSQL_URL="mysql+pymysql://${MYSQL_USER}:@${MYSQL_HOST}:${MYSQL_PORT}/${MYSQL_DB}"

echo "Updating index weight (dump CSV and import to MySQL)"
# Determine start date from existing table
PASS_OPT=""
if [[ -n "${MYSQL_PASSWORD}" ]]; then PASS_OPT="-p${MYSQL_PASSWORD}"; fi
IDX_WEIGHT_START=$(mysql -h "${MYSQL_HOST}" -P "${MYSQL_PORT}" -u"${MYSQL_USER}" --protocol=tcp ${PASS_OPT} -N -s -e \
  "SELECT IFNULL(DATE_FORMAT(DATE_ADD(MAX(trade_date), INTERVAL 1 DAY),'%Y%m%d'),'19900101') FROM ${MYSQL_DB}.ts_index_weight" 2>/dev/null || echo "19900101")
python3 "${SCRIPT_DIR}/tushare_provider/dump_index_weight.py" --start_date="${IDX_WEIGHT_START}"

# Import all CSVs under tushare_provider/index_weight into ts_index_weight
python3 "${SCRIPT_DIR}/tushare_provider/import_index_weight_mysql.py" \
  --mysql_url="${MYSQL_URL}" \
  --csv_dir="${SCRIPT_DIR}/tushare_provider/index_weight" \
  --chunksize=2000

echo "Updating index price (dump CSV and import to MySQL)"
# Compute start date from existing index symbols in ts_a_stock_eod_price
INDEX_LIST="('399300.SZ','000905.SH','000300.SH','000906.SH','000852.SH','000985.SH')"
IDX_PRICE_START=$(mysql -h "${MYSQL_HOST}" -P "${MYSQL_PORT}" -u"${MYSQL_USER}" --protocol=tcp ${PASS_OPT} -N -s -e \
  "SELECT IFNULL(DATE_FORMAT(DATE_ADD(MAX(tradedate), INTERVAL 1 DAY),'%Y%m%d'),'19900101') FROM ${MYSQL_DB}.ts_a_stock_eod_price WHERE symbol IN ${INDEX_LIST}" 2>/dev/null || echo "19900101")
python3 "${SCRIPT_DIR}/tushare_provider/dump_index_eod_price.py" --start_date="${IDX_PRICE_START}"

# Import index CSVs into ts_a_stock_eod_price
python3 "${SCRIPT_DIR}/tushare_provider/import_index_price_mysql.py" \
  --mysql_url="${MYSQL_URL}" \
  --csv_dir="${SCRIPT_DIR}/tushare_provider/index" \
  --chunksize=2000

echo "Updating A-share EOD price (direct MySQL)"
python3 "${SCRIPT_DIR}/tushare_provider/update_a_stock_eod_price_to_latest.py" 

echo "Updating fundamentals (direct MySQL)"
# Note: update_a_stock_fundamental.py currently connects to 127.0.0.1:3306/investment_data by default.
# If your MySQL runs elsewhere, adjust that script or ensure a port forward to 3306.
START_ARG=()
if [[ -n "${START_DATE_OVERRIDE}" ]]; then
  START_ARG=("--start_date_override=${START_DATE_OVERRIDE}")
fi
python3 "${SCRIPT_DIR}/tushare_provider/update_a_stock_fundamental.py" "${START_ARG[@]}"

echo "Updating A-share basic info"
python3 "${SCRIPT_DIR}/tushare_provider/update_a_stock_basic.py" --mysql_url="${MYSQL_URL}" 

echo "Updating A-share financial profile"
python3 "${SCRIPT_DIR}/tushare_provider/update_a_stock_financial_profile.py" "20200101" --mysql_url="${MYSQL_URL}" --period quarter

echo "Updating A-share suspend info"
python3 "${SCRIPT_DIR}/tushare_provider/update_a_stock_suspend_info.py" --mysql_url="${MYSQL_URL}"

echo "Updating A-share cost percentage"
python3 "${SCRIPT_DIR}/tushare_provider/update_a_stock_cost_pct.py" --mysql_url="${MYSQL_URL}"

echo "Updating A-share moenyflow"
python3 "${SCRIPT_DIR}/tushare_provider/update_a_stock_moneyflow.py" --mysql_url="${MYSQL_URL}"

echo "Updating A-share brokerage report"
python3 "${SCRIPT_DIR}/tushare_provider/update_a_stock_brokerage_report.py" --mysql_url="${MYSQL_URL}"

echo "evaluate A-share brokerage report"
python3 "${SCRIPT_DIR}/tushare_provider/evaluate_brokerage_report.py" --mysql_url="${MYSQL_URL}"

echo "Merging to final tables"
mysql -h "${MYSQL_HOST}" -P "${MYSQL_PORT}" -u"${MYSQL_USER}" --protocol=tcp ${PASS_OPT} "${MYSQL_DB}" < "${SCRIPT_DIR}/sql/price.sql" 
mysql -h "${MYSQL_HOST}" -P "${MYSQL_PORT}" -u"${MYSQL_USER}" --protocol=tcp ${PASS_OPT} "${MYSQL_DB}" < "${SCRIPT_DIR}/sql/fundamental.sql" &
mysql -h "${MYSQL_HOST}" -P "${MYSQL_PORT}" -u"${MYSQL_USER}" --protocol=tcp ${PASS_OPT} "${MYSQL_DB}" < "${SCRIPT_DIR}/sql/moneyflow.sql" &
mysql -h "${MYSQL_HOST}" -P "${MYSQL_PORT}" -u"${MYSQL_USER}" --protocol=tcp ${PASS_OPT} "${MYSQL_DB}" < "${SCRIPT_DIR}/sql/cost.sql" &
mysql -h "${MYSQL_HOST}" -P "${MYSQL_PORT}" -u"${MYSQL_USER}" --protocol=tcp ${PASS_OPT} "${MYSQL_DB}" < "${SCRIPT_DIR}/sql/brokerage.sql" &

#wait for all finished
wait

echo "Daily MySQL update finished."
