import os
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime
from functools import wraps
import time
import logging

def setup_logging(level=logging.INFO, format_str='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', log_file=None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]

    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers
    )

def init_tushare(token=None):
    """Initialize Tushare API with token"""
    if token is None:
        token = os.environ.get("TUSHARE")

    if not token:
        raise ValueError("TUSHARE environment variable not set or token not provided. Please set your Tushare token.")

    ts.set_token(token)
    return ts.pro_api()

class CacheManager:
    """Cache manager for data persistence"""

    def __init__(self, cache_dir="cache", use_cache=True):
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        if self.use_cache and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, cache_file: str) -> str:
        """Get the full path for a cache file"""
        return os.path.join(self.cache_dir, f"{cache_file}.parquet")

    def load_from_cache(self, cache_file: str) -> pd.DataFrame:
        """Load data from cache if it exists"""
        if not self.use_cache:
            return pd.DataFrame()

        cache_path = self.get_cache_path(cache_file)
        if os.path.exists(cache_path):
            try:
                logger.info(f"Loading {cache_file} from cache: {cache_path}")
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_file}: {e}")
        return pd.DataFrame()

    def save_to_cache(self, file_name: str, df: pd.DataFrame):
        """Save data to cache"""
        if not self.use_cache or df.empty:
            return

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        cache_path = self.get_cache_path(file_name)
        try:
            df.to_parquet(cache_path, index=False)
            logger.info(f"Saved to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator: Add retry mechanism to function

    Args:
        max_retries: Maximum number of retries
        delay: Initial delay time (seconds)
        backoff: Delay multiplier
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < max_retries:
                        logger.warning(f"Function {func.__name__} call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        logger.info(f"Waiting {current_delay:.1f} seconds before retry...")
                        time.sleep(current_delay)
                        current_delay *= backoff  # exponential backoff
                    else:
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise last_exception

            # This line won't be executed, but for type checker
            raise last_exception

        return wrapper
    return decorator


def call_tushare_api_with_retry(api_func, *args, **kwargs):
    """
    Generic Tushare API call function with retry mechanism

    Args:
        api_func: Tushare API function
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        DataFrame returned by API
    """
    @retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
    def _call_api():
        return api_func(*args, **kwargs)

    return _call_api()

