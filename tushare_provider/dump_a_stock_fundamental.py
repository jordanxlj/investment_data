import tushare as ts
import os
import pandas
import fire
import time


ts.set_token(os.environ["TUSHARE"])
pro = ts.pro_api()
file_path = os.path.dirname(os.path.realpath(__file__))


def get_trade_cal(start_date: str, end_date: str) -> pandas.DataFrame:
    df = pro.trade_cal(
        exchange="SSE",
        is_open="1",
        start_date=start_date,
        end_date=end_date,
        fields="cal_date",
    )
    return df


def get_daily_basic(trade_date: str) -> pandas.DataFrame | None:
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


def dump_astock_fundamental(start_date: str = "19900101", end_date: str = "20500101", skip_exists: bool = True):
    trade_date_df = get_trade_cal(start_date, end_date)
    # Ensure output dir exists
    output_dir = f"{file_path}/astock_fundamental"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate dates ascending
    trade_date_df = trade_date_df.sort_values("cal_date").reset_index(drop=True)
    for row in trade_date_df.values.tolist():
        trade_date = row[0]
        filename = f"{output_dir}/{trade_date}.csv"
        print(filename)
        if skip_exists and os.path.isfile(filename):
            continue
        data = get_daily_basic(trade_date)
        if data is None:
            continue
        if data.empty:
            continue
        data.to_csv(filename, index=False)


if __name__ == "__main__":
    fire.Fire(dump_astock_fundamental)


