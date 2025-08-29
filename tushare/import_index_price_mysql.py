import os
import pandas as pd
import fire
from sqlalchemy import create_engine
import pymysql  # noqa: F401


def _map_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    if 'tradedate' in df.columns:
        out['tradedate'] = pd.to_datetime(df['tradedate']).dt.strftime('%Y-%m-%d')
    elif 'trade_date' in df.columns:
        out['tradedate'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
    else:
        return pd.DataFrame()
    if 'symbol' in df.columns:
        out['symbol'] = df['symbol']
    elif 'ts_code' in df.columns:
        out['symbol'] = df['ts_code']
    else:
        return pd.DataFrame()
    for c in ['high','low','open','close']:
        out[c] = df[c] if c in df.columns else None
    if 'volume' in df.columns:
        out['volume'] = df['volume']
    elif 'vol' in df.columns:
        out['volume'] = df['vol']
    else:
        out['volume'] = None
    out['amount'] = df['amount'] if 'amount' in df.columns else None
    if 'adjclose' in df.columns:
        out['adjclose'] = df['adjclose']
    else:
        out['adjclose'] = df['close'] if 'close' in df.columns else None
    return out[['tradedate','symbol','high','low','open','close','volume','adjclose','amount']]


def import_index_price_mysql(
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
            mapped = _map_frame(df)
            if mapped.empty:
                print('skip (no required columns):', path)
                continue
            mapped.to_sql('ts_a_stock_eod_price', engine, if_exists='append', index=False, method='multi', chunksize=chunksize)
            print('imported', path)
        except Exception as e:
            print('index price import failed for', path, e)


if __name__ == '__main__':
    fire.Fire(import_index_price_mysql)


