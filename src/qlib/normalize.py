import fire
import pandas as pd
import os

try:
  from data_collector.base import Normalize
  from data_collector.yahoo import collector as yahoo_collector
except ImportError as e:
  print("============")
  print("ATTENTION: Need to put qlib/scripts directory into PYTHONPATH")
  print("============")
  raise e

class CrowdSourceNormalize(yahoo_collector.YahooNormalizeCN1d):
  # Add vwap so that vwap will be adjusted during normalization
  COLUMNS = ["open", "close", "high", "low", "vwap", "volume"]
  EXCLUDES = ['amount', 'turnover', 'volume_ratio', 'dividend_ratio', 'pe', 
              'pb', 'ps', 'market_cap', 'main_inflow_ratio', 'small_inflow_ratio', 
              'net_inflow_ratio', 'cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct', 
              'cost_95pct', 'weight_avg', 'winner_rate', 'f_pos_ratio', 'f_neg_ratio', 
              'f_eps', 'f_pe', 'f_dv_ratio', 'f_roe', 'current_ratio', 'quick_ratio', 
              'cash_ratio', 'ca_turn', 'inv_turn', 'ar_turn', 'fa_turn', 'assets_turn', 
              'roic', 'roe_ttm', 'roa_ttm', 'grossprofit_margin_ttm', 'netprofit_margin_ttm', 
              'fcf_margin_ttm', 'debt_to_assets', 'debt_to_eqt', 'debt_to_ebitda', 
              'bps', 'eps_ttm', 'revenue_ps_ttm', 'cfps', 'fcff_ps', 'or_yoy', 
              'netprofit_yoy', 'basic_eps_yoy', 'equity_yoy', 'assets_yoy', 'ocf_yoy', 
              'roe_yoy', 'revenue_cagr_3y', 'netincome_cagr_3y', 'rd_exp_to_capex', 
              'goodwill']
  def _manual_adj_data(self, df: pd.DataFrame) -> pd.DataFrame:
    # amount should be kept as original value, so that adjusted volume * adjust vwap = amount
    result_df = super()._manual_adj_data(df)
    for column in self.EXCLUDES:
        if column in result_df.columns and column in df.columns:
            result_df[column] = df[column]
    return result_df

def _clean_empty_files(directory: str) -> None:
    if not directory or not os.path.isdir(directory):
        return
    removed = 0
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        try:
            if os.path.getsize(path) == 0:
                os.remove(path)
                removed += 1
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                chunk = f.read(256)
                if not chunk.strip():
                    os.remove(path)
                    removed += 1
        except Exception:
            # ignore unreadable files
            continue
    if removed:
        print(f"Pre-clean: removed {removed} empty files from {directory}")


def normalize_crowd_source_data(source_dir=None, normalize_dir=None, max_workers=1, interval="1d", date_field_name="tradedate", symbol_field_name="symbol", preclean_empty: bool = True):
    if preclean_empty and source_dir:
        _clean_empty_files(source_dir)
    yc = Normalize(
        source_dir=source_dir,
        target_dir=normalize_dir,
        normalize_class=CrowdSourceNormalize,
        max_workers=max_workers,
        date_field_name=date_field_name,
        symbol_field_name=symbol_field_name,
    )
    yc.normalize()

if __name__ == "__main__":
    fire.Fire(normalize_crowd_source_data)