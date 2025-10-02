set -e
set -x
WORKING_DIR=/workspace

MYSQL_HOST=${MYSQL_HOST:-127.0.0.1}
MYSQL_PORT=${MYSQL_PORT:-3307}
MYSQL_USER=${MYSQL_USER:-root}
MYSQL_PASSWORD=${MYSQL_PASSWORD:-}
MYSQL_DB=${MYSQL_DB:-investment_data_new}
MYSQL_URL="mysql+pymysql://${MYSQL_USER}:@${MYSQL_HOST}:${MYSQL_PORT}/${MYSQL_DB}"
QLIB_DATA="./temp_dir/qlib_source"
NORMALIZE_DIR="./temp_dir/qlib_normalize"
INDEX_DIR="./temp_dir/qlib_index"
SCRIPT_DIR="./src"

mkdir -p $QLIB_DATA
python3 ./src/qlib/dump_all_to_qlib_source.py --mysql_url="${MYSQL_URL}" --output_dir=$QLIB_DATA

export PYTHONPATH=`pwd`
export PYTHONPATH=$PYTHONPATH:$WORKING_DIR/qlib_enhanced:$WORKING_DIR/qlib_enhanced/scripts

python3 $SCRIPT_DIR/qlib/normalize.py normalize_data --source_dir $QLIB_DATA --normalize_dir $NORMALIZE_DIR --max_workers=16 --date_field_name="tradedate"
python3 $WORKING_DIR/qlib_enhanced/scripts/dump_bin.py dump_all --csv_path $NORMALIZE_DIR --qlib_dir $WORKING_DIR/qlib_bin --date_field_name=tradedate --exclude_fields=tradedate,symbol

mkdir -p $INDEX_DIR
python3 $SCRIPT_DIR/qlib/dump_index_weight.py --mysql_url="${MYSQL_URL}"
python3 $SCRIPT_DIR/tushare_provider/dump_day_calendar.py --mysql_url="${MYSQL_URL}" --qlib_dir=$WORKING_DIR/qlib_bin/

cp $INDEX_DIR/csi* $WORKING_DIR/qlib_bin/instruments/

tar -czvf ./qlib_bin.tar.gz $WORKING_DIR/qlib_bin/
ls -lh ./qlib_bin.tar.gz
echo "Generated tarball at $(pwd)/qlib_bin.tar.gz"
