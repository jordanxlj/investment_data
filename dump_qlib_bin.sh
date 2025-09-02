set -e
set -x
WORKING_DIR=/workspace

MYSQL_HOST=${MYSQL_HOST:-127.0.0.1}
MYSQL_PORT=${MYSQL_PORT:-3307}
MYSQL_USER=${MYSQL_USER:-root}
MYSQL_PASSWORD=${MYSQL_PASSWORD:-}
MYSQL_DB=${MYSQL_DB:-investment_data_new}
MYSQL_URL="mysql+pymysql://${MYSQL_USER}:@${MYSQL_HOST}:${MYSQL_PORT}/${MYSQL_DB}"

mkdir -p ./qlib/qlib_source
python3 ./qlib/dump_all_to_qlib_source.py --mysql_url="${MYSQL_URL}"

export PYTHONPATH=`pwd`
export PYTHONPATH=$PYTHONPATH:$WORKING_DIR/qlib_enhanced:$WORKING_DIR/qlib_enhanced/scripts

python3 ./qlib/normalize.py normalize_data --source_dir ./qlib/qlib_source/ --normalize_dir ./qlib/qlib_normalize --max_workers=16 --date_field_name="tradedate"
python3 $WORKING_DIR/qlib_enhanced/scripts/dump_bin.py dump_all --csv_path ./qlib/qlib_normalize/ --qlib_dir $WORKING_DIR/qlib_bin --date_field_name=tradedate --exclude_fields=tradedate,symbol

mkdir -p ./qlib/qlib_index/
python3 ./qlib/dump_index_weight.py --mysql_url="${MYSQL_URL}"
python3 ./tushare/dump_day_calendar.py --mysql_url="${MYSQL_URL}" --qlib_dir=$WORKING_DIR/qlib_bin/

cp qlib/qlib_index/csi* $WORKING_DIR/qlib_bin/instruments/

tar -czvf ./qlib_bin.tar.gz $WORKING_DIR/qlib_bin/
ls -lh ./qlib_bin.tar.gz
echo "Generated tarball at $(pwd)/qlib_bin.tar.gz"
