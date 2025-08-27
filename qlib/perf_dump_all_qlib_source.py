import os
import csv
import datetime as dt
from typing import Optional, Tuple

import pymysql
import fire


def _next(cur) -> Optional[Tuple]:
    try:
        return cur.fetchone()
    except Exception:
        return None


def _cmp_key(a: Tuple, b: Tuple) -> int:
    if a[0] < b[0]:
        return -1
    if a[0] > b[0]:
        return 1
    if a[1] < b[1]:
        return -1
    if a[1] > b[1]:
        return 1
    return 0


def perf_dump_all_qlib_source(
    skip_exists: bool = True,
    start_date: Optional[str] = None,  # YYYYMMDD
    end_date: Optional[str] = None,    # YYYYMMDD
    output_dir: Optional[str] = None,
    mysql_host: str = "127.0.0.1",
    mysql_user: str = "root",
    mysql_password: str = "",
    mysql_db: str = "investment_data",
    fetch_rows: int = 10000,
):
    """
    High-performance dump without DB-side big joins:
    - Streams price and fundamental tables with server-side cursors, ordered by (symbol, tradedate)
    - Two-way merge in Python (O(N+M) time, O(1) memory)
    - Writes per-symbol CSV incrementally to avoid large in-memory frames
    """

    if output_dir is None:
        script_path = os.path.dirname(os.path.realpath(__file__))
        output_dir = f"{script_path}/qlib_source"
    os.makedirs(output_dir, exist_ok=True)

    date_cond = []
    if start_date:
        date_cond.append(f"tradedate >= STR_TO_DATE('{start_date}', '%Y%m%d')")
    if end_date:
        date_cond.append(f"tradedate <= STR_TO_DATE('{end_date}', '%Y%m%d')")
    where_clause = (" WHERE " + " AND ".join(date_cond)) if date_cond else ""

    price_sql = (
        "SELECT symbol, tradedate, high, low, open, close, volume, amount, adjclose "
        f"FROM final_a_stock_eod_price{where_clause} "
        "ORDER BY symbol, tradedate"
    )

    fund_sql = (
        "SELECT symbol, tradedate, turnover_rate_f, volume_ratio, pe_ttm, pb, ps_ttm, dv_ttm, float_share, circ_mv "
        f"FROM final_a_stock_fundamental{where_clause} "
        "ORDER BY symbol, tradedate"
    )

    conn = pymysql.connect(
        host=mysql_host,
        user=mysql_user,
        password=mysql_password,
        database=mysql_db,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.SSCursor,  # server-side streaming
    )

    print(f"connected sqlserver successfully...")
    price_cur = conn.cursor()
    fund_cur = conn.cursor()
    print(f"execute {price_sql}")
    price_cur.execute(price_sql)
    print(f"execute {fund_sql}")
    fund_cur.execute(fund_sql)

    price_row = _next(price_cur)
    fund_row = _next(fund_cur)

    current_symbol = None
    writer = None
    file_handle = None
    skip_symbol = False
    cutoff_for_symbol: Optional[dt.date] = None

    def _read_last_tradedate_from_csv(path: str) -> Optional[dt.date]:
        if not os.path.isfile(path):
            return None
        try:
            last = None
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        last = line.strip()
            if not last:
                return None
            first = last.split(",", 1)[0]
            return dt.datetime.strptime(first, "%Y-%m-%d").date()
        except Exception:
            return None

    def ensure_writer(symbol: str) -> Optional[dt.date]:
        nonlocal current_symbol, writer, file_handle
        if symbol == current_symbol:
            return None
        # close previous
        if file_handle:
            file_handle.flush()
            file_handle.close()
        current_symbol = symbol
        filename = os.path.join(output_dir, f"{symbol}.csv")
        cutoff: Optional[dt.date] = None
        if os.path.isfile(filename):
            cutoff = _read_last_tradedate_from_csv(filename)
            file_handle = open(filename, "a", newline="", encoding="utf-8")
            writer = csv.writer(file_handle)
        else:
            file_handle = open(filename, "w", newline="", encoding="utf-8")
            writer = csv.writer(file_handle)
            writer.writerow([
                "tradedate","symbol","high","low","open","close","volume","amount","adjclose","vwap",
                "turnover","volume_ratio","pe","pb","ps","dividend_ratio","float_share","market_cap"
            ])
        return cutoff

    # Merge loop
    rows_written = 0
    print(f"update the csv file by the symbol and date...")
    while price_row is not None:
        p_symbol, p_date, p_high, p_low, p_open, p_close, p_vol, p_amt, p_adj = price_row

        # advance fundamentals until >= price key
        while fund_row is not None and _cmp_key((fund_row[0], fund_row[1]), (p_symbol, p_date)) < 0:
            fund_row = _next(fund_cur)

        # match?
        f_vals = (None,)*8
        if fund_row is not None and _cmp_key((fund_row[0], fund_row[1]), (p_symbol, p_date)) == 0:
            _, _, f_turnover, f_volr, f_pe, f_pb, f_ps, f_div, f_float, f_mcap = fund_row
            f_vals = (f_turnover, f_volr, f_pe, f_pb, f_ps, f_div, f_float, f_mcap)

        # write row
        new_cutoff = ensure_writer(p_symbol)
        if new_cutoff is not None:
            cutoff_for_symbol = new_cutoff
        if writer:
            if not cutoff_for_symbol or p_date > cutoff_for_symbol:
                vwap = (p_amt / p_vol * 10.0) if (p_vol and p_vol != 0) else None
                writer.writerow([
                    p_date.strftime("%Y-%m-%d"), p_symbol, p_high, p_low, p_open, p_close, p_vol, p_amt, p_adj, vwap,
                    *f_vals
                ])
                rows_written += 1

        # advance price
        price_row = _next(price_cur)

        # periodic flush
        if file_handle and (rows_written % fetch_rows == 0):
            file_handle.flush()

    # cleanup
    if file_handle:
        file_handle.flush()
        file_handle.close()
    price_cur.close()
    fund_cur.close()
    conn.close()


if __name__ == "__main__":
    fire.Fire(perf_dump_all_qlib_source)


