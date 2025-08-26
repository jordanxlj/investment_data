import os
import time
import datetime
from typing import Optional

import fire
import pandas
import tushare as ts
from sqlalchemy import create_engine
import pymysql  # noqa: F401 - required by SQLAlchemy URL


ts.set_token(os.environ["TUSHARE"])
pro = ts.pro_api()


def get_trade_cal(start_date: str, end_date: str) -> pandas.DataFrame:
    df = pro.trade_cal(
        exchange="SSE",
        is_open="1",
        start_date=start_date,
        end_date=end_date,
        fields="cal_date",
    )
    return df


def get_daily_basic(trade_date: str) -> Optional[pandas.DataFrame]:
    fields = (
        "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,"
        "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,"
        "total_share,float_share,free_share,total_mv,circ_mv"
    )
    for _ in range(3):
        try:
            df = pro.daily_basic(trade_date=trade_date, fields=fields)
            return df
        except Exception as e:
            print(e)
            time.sleep(1)
    return None


def update_astock_fundamental_to_latest() -> None:
    """Append missing ts_a_stock_fundamental rows to MySQL from Tushare daily_basic.

    - Finds latest trade_date present in table
    - Iterates subsequent open days up to today
    - Fetches daily_basic and appends to table
    """

    sql_engine = create_engine(
        "mysql+pymysql://root:@127.0.0.1/investment_data", pool_recycle=3600
    )
    db_conn = sql_engine.raw_connection()

    # Determine latest fully populated trade_date present
    latest_sql = (
        """
        select max(trade_date) as trade_date
        from (
            select trade_date, count(1) as symbol_count
            from ts_a_stock_fundamental
            group by trade_date
        ) t
        where symbol_count > 1000
        """
    )

    latest_trade_date: Optional[str] = None
    try:
        latest_df = pandas.read_sql(latest_sql, db_conn)
        val = latest_df["trade_date"][0]
        if pandas.notna(val):
            latest_trade_date = str(val)
    except Exception as e:
        print("Failed to read latest trade_date:", e)

    if not latest_trade_date:
        latest_trade_date = "19900101"

    end_date = datetime.datetime.now().strftime("%Y%m%d")

    trade_date_df = get_trade_cal(latest_trade_date, end_date)
    trade_date_df = trade_date_df.sort_values("cal_date").reset_index(drop=True)

    for row in trade_date_df.values.tolist():
        trade_date = row[0]
        if trade_date == latest_trade_date:
            continue
        print("Downloading fundamentals", trade_date)
        data = get_daily_basic(trade_date)
        if data is None or data.empty:
            continue
        # Persist
        record_num = data.to_sql(
            "ts_a_stock_fundamental", sql_engine, if_exists="append", index=False
        )
        print(f"{trade_date} Updated fundamentals: {record_num} records")


if __name__ == "__main__":
    fire.Fire(update_astock_fundamental_to_latest)


