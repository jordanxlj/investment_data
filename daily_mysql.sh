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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MYSQL_URL="mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@${MYSQL_HOST}:${MYSQL_PORT}/${MYSQL_DB}"

echo "Updating fundamentals (direct MySQL)"
# Note: update_a_stock_fundamental.py currently connects to 127.0.0.1:3306/investment_data by default.
# If your MySQL runs elsewhere, adjust that script or ensure a port forward to 3306.
START_ARG=()
if [[ -n "${START_DATE_OVERRIDE}" ]]; then
  START_ARG=("--start_date_override=${START_DATE_OVERRIDE}")
fi
python3 "${SCRIPT_DIR}/tushare/update_a_stock_fundamental.py" "${START_ARG[@]}"

echo "Updating A-share EOD price (direct MySQL)"
python3 "${SCRIPT_DIR}/tushare/update_a_stock_eod_price_to_latest.py" 

echo "Merging to final tables"
PASS_OPT=""
if [[ -n "${MYSQL_PASSWORD}" ]]; then PASS_OPT="-p${MYSQL_PASSWORD}"; fi
mysql -h "${MYSQL_HOST}" -P "${MYSQL_PORT}" -u"${MYSQL_USER}" --protocol=tcp ${PASS_OPT} "${MYSQL_DB}" < "${SCRIPT_DIR}/tushare/regular_update.sql"

echo "Daily MySQL update finished."
