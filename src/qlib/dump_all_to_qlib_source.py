from sqlalchemy import create_engine
import pymysql
import pandas as pd
import fire
import os

def dump_all_to_qlib_source(
        mysql_url='mysql+pymysql://root:@127.0.0.1/investment_data',
        output_dir='temp_dir/qlib_source',
        skip_exists=True
    ):
    """Export final_a_stock_comb_info to parquet format with gzip compression"""

    sqlEngine = create_engine(mysql_url, pool_recycle=3600)
    dbConnection = sqlEngine.raw_connection()

    # Export all fields from final_a_stock_comb_info
    chunks = pd.read_sql(
        """
        SELECT
            tradedate,
            symbol,
            industry,
            high,
            low,
            open,
            close,
            volume,
            adjclose,
            amount,
            amount/volume*10 as vwap,
            turnover,
            volume_ratio,
            pe,
            pb,
            ps,
            dv_ratio,
            circ_mv as market_cap,
            main_inflow_ratio,
            small_inflow_ratio,
            net_inflow_ratio,
            cost_5pct,
            cost_15pct,
            cost_50pct,
            cost_85pct,
            cost_95pct,
            weight_avg,
            winner_rate,
            f_pos_ratio,
            f_neg_ratio,
            f_target_price,
            f_eps,
            f_pe,
            f_dv_ratio,
            f_roe,
            -- Liquidity ratios
            current_ratio,
            quick_ratio,
            cash_ratio,
            -- Turnover ratios
            ca_turn,
            inv_turn,
            ar_turn,
            fa_turn,
            assets_turn,
            -- Profitability ratios
            roic,
            roe_ttm,
            roa_ttm,
            grossprofit_margin_ttm,
            netprofit_margin_ttm,
            fcf_margin_ttm,
            -- Leverage ratios
            debt_to_assets,
            debt_to_eqt,
            debt_to_ebitda,
            -- Valuation metrics
            bps,
            eps_ttm,
            revenue_ps_ttm,
            cfps,
            fcff_ps,
            -- Growth metrics
            or_yoy,
            netprofit_yoy,
            basic_eps_yoy,
            equity_yoy,
            assets_yoy,
            ocf_yoy,
            roe_yoy,
            revenue_cagr_3y,
            netincome_cagr_3y,
            -- Other
            rd_exp_to_capex,
            goodwill
        FROM final_a_stock_comb_info
        where tradedate >= '2018-01-01'
        ORDER BY symbol, tradedate
        """,
        dbConnection,
        chunksize=100000
    )

    stock_df = pd.DataFrame()
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i} with {len(chunk)} rows")
        stock_df = pd.concat([stock_df, chunk], ignore_index=True)

    stock_df.sort_values(['symbol', 'tradedate'], inplace=True)

    dbConnection.close()
    sqlEngine.dispose()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Group by symbol and save each as parquet with gzip compression
    for symbol, df in stock_df.groupby("symbol"):
        filename = f'{output_dir}/{symbol}.parquet'
        print(f"Dumping to parquet file: {filename}")

        if skip_exists and os.path.isfile(filename):
            print(f"Skipping existing file: {filename}")
            continue

        # Save as parquet with gzip compression
        df.to_parquet(filename, index=False, compression='gzip')

    print(f"Export completed. Processed {len(stock_df)} total records for {stock_df['symbol'].nunique()} symbols")

if __name__ == "__main__":
    fire.Fire(dump_all_to_qlib_source)
