from sqlalchemy import create_engine
import pymysql
import pandas as pd
import fire
import os

def dump_all_to_sqlib_source(skip_exists=True):
  sqlEngine = create_engine('mysql+pymysql://root:@127.0.0.1/investment_data', pool_recycle=3600)
  dbConnection = sqlEngine.raw_connection()
  stock_df = pd.read_sql(
    """
    select 
      p.*, 
      p.amount/p.volume*10 as vwap,
      f.turnover_rate_f as turnover,
      f.volume_ratio,
      f.pe_ttm as pe,
      f.pb,
      f.ps_ttm as ps,
      f.dv_ttm as dividend_ratio,
      f.float_share,
      f.circ_mv as market_cap
    from final_a_stock_eod_price p
    left join final_a_stock_fundamental f
      on f.tradedate = p.tradedate and f.symbol = p.symbol
    """,
    dbConnection,
  )
  dbConnection.close()
  sqlEngine.dispose()

  script_path = os.path.dirname(os.path.realpath(__file__))
  output_dir = f"{script_path}/qlib_source"
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  for symbol, df in stock_df.groupby("symbol"):
    filename = f'{output_dir}/{symbol}.csv'
    print("Dumping to file: ", filename)
    if skip_exists and os.path.isfile(filename):
        continue
    df.to_csv(filename, index=False)

if __name__ == "__main__":
  fire.Fire(dump_all_to_sqlib_source)
