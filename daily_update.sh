set -e
set -x

[ ! -d "/dolt/investment_data" ] && echo "initializing dolt repo" && cd /dolt && dolt clone chenditc/investment_data
cd /dolt/investment_data
dolt pull
dolt push

echo "Updating index weight"
startdate=$(dolt sql -q "select * from max_index_date" -r csv | tail -1)
python3 /investment_data/tushare/dump_index_weight.py --start_date=$startdate
for file in $(ls /investment_data/tushare/index_weight/); 
do  
  dolt table import -u ts_index_weight /investment_data/tushare/index_weight/$file; 
done

echo "Updating index price"
python3 /investment_data/tushare/dump_index_eod_price.py 
for file in $(ls /investment_data/tushare/index/); 
do   
  dolt table import -u ts_a_stock_eod_price /investment_data/tushare/index/$file; 
done

echo "Updating fundamentals"
# Try to get last date from fundamentals table; fallback if missing
fund_startdate=$(dolt sql -q "select date_format(max(trade_date), '%Y%m%d') from ts_a_stock_fundamental" -r csv 2>/dev/null || true)
fund_startdate=$(echo "$fund_startdate" | tail -1)
if [ -z "$fund_startdate" ] || [ "$fund_startdate" = "NULL" ]; then fund_startdate=19900101; fi
python3 /investment_data/tushare/dump_a_stock_fundamental.py --start_date=$fund_startdate
for file in $(ls /investment_data/tushare/astock_fundamental/); 
do  
  dolt table import -u -pk ts_code,trade_date ts_a_stock_fundamental /investment_data/tushare/astock_fundamental/$file; 
done

echo "Updating stock price"
dolt sql-server &
sleep 5 && python3 /investment_data/tushare/update_a_stock_eod_price_to_latest.py
killall dolt

dolt sql --file /investment_data/tushare/regular_update.sql

dolt add -A

status_output=$(dolt status)

# Check if the status output contains the "nothing to commit, working tree clean" message
if [[ $status_output == *"nothing to commit, working tree clean"* ]]; then
    echo "No changes to commit. Working tree is clean."
else
    echo "Changes found. Committing and pushing..."
    # Run the necessary commands
    dolt commit -m "Daily update"
    dolt push 
    echo "Changes committed and pushed."
fi

