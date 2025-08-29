import os
import pandas as pd
import fire
from sqlalchemy import create_engine
import pymysql  # noqa: F401


def import_index_weight_mysql(
    mysql_url: str,
    csv_dir: str,
    chunksize: int = 2000,
):
    engine = create_engine(mysql_url, pool_recycle=3600)
    if not os.path.isdir(csv_dir):
        print("csv_dir not found:", csv_dir)
        return
    files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]
    for path in files:
        try:
            df = pd.read_csv(path)
            cols = list(df.columns)
            needed = ['index_code','stock_code','trade_date','weight']
            missing = [c for c in needed if c not in cols]
            if missing:
                print('skip (missing cols):', path, missing)
                continue
            # Map to destination schema: index_code, stock_code, trade_date(YYYY-MM-DD), weight
            out = pd.DataFrame()
            out['index_code'] = df['index_code']
            out['stock_code'] = df['stock_code']
            out['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce').dt.strftime('%Y-%m-%d')
            out['weight'] = df['weight']
            out = out.dropna(subset=['index_code','stock_code','trade_date'])
            if out.empty:
                print('skip (no valid rows after mapping):', path)
                continue
            out[['index_code','stock_code','trade_date','weight']].to_sql(
                'ts_index_weight', engine, if_exists='append', index=False, method='multi', chunksize=chunksize
            )
            print('imported', path)
        except Exception as e:
            print('index_weight import failed for', path, e)


if __name__ == '__main__':
    fire.Fire(import_index_weight_mysql)


