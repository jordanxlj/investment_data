@echo on
setlocal EnableExtensions EnableDelayedExpansion

:: Configuration
set "SCRIPT_DIR=%~dp0"
set "INVESTMENT_DATA_DIR=%SCRIPT_DIR%"
set "DOLT_ROOT=d:\code\stock\dolt"
if "%DOLT_REPO_DIR%"=="" set "DOLT_REPO_DIR=%DOLT_ROOT%\investment_data"
set "PYTHON=python"
set "TUSHARE=362c632d025da65bb9b5bd8e6bcaf9ab2927947043f3d051effa8673"

:: Ensure dolt repo exists
if not exist "%DOLT_REPO_DIR%" (
  echo initializing dolt repo
  if not exist "%DOLT_ROOT%" mkdir "%DOLT_ROOT%"
  pushd "%DOLT_ROOT%"
  dolt clone chenditc/investment_data
  if errorlevel 1 (
    echo Failed to clone dolt repo
    popd
    exit /b 1
  )
  popd
)

:: Navigate to dolt repo
pushd "%DOLT_REPO_DIR%"

:: Sync latest changes
::dolt pull origin master || exit /b 1
::dolt push

echo Updating index weight
set "startdate="
for /f "skip=1 delims=" %%A in ('dolt sql -q "select * from max_index_date" -r csv') do set "startdate=%%A"
echo startdate=!startdate!
::"%PYTHON%" "%INVESTMENT_DATA_DIR%tushare\dump_index_weight.py" --start_date=!startdate!
for %%F in ("%INVESTMENT_DATA_DIR%tushare\index_weight\*.*") do (
  dolt table import -u ts_index_weight "%%~fF"
)

echo Updating index price
::"%PYTHON%" "%INVESTMENT_DATA_DIR%tushare\dump_index_eod_price.py"
for %%F in ("%INVESTMENT_DATA_DIR%tushare\index\*.*") do (
  dolt table import -u ts_a_stock_eod_price "%%~fF"
)

echo Updating stock price
start "dolt-sql-server" dolt sql-server
timeout /t 5 /nobreak >nul
::"%PYTHON%" "%INVESTMENT_DATA_DIR%tushare\update_a_stock_eod_price_to_latest.py"
taskkill /IM dolt.exe /F >nul 2>&1

echo Updating fundamentals
set "fund_startdate="
for /f "skip=1 delims=" %%A in ('dolt sql -q "select date_format(max(trade_date), '%%Y%%m%%d') from ts_a_stock_fundamental" -r csv 2^>nul') do set "fund_startdate=%%A"
if "%fund_startdate%"=="" set "fund_startdate=20000101"
if /I "%fund_startdate%"=="NULL" set "fund_startdate=20000101"

:: ensure ts_a_stock_fundamental exists
dolt sql -q "CREATE TABLE IF NOT EXISTS ts_a_stock_fundamental ( ts_code VARCHAR(16) NOT NULL, trade_date VARCHAR(8) NOT NULL, turnover_rate DOUBLE, turnover_rate_f DOUBLE, volume_ratio DOUBLE, pe DOUBLE, pe_ttm DOUBLE, pb DOUBLE, ps DOUBLE, ps_ttm DOUBLE, dv_ratio DOUBLE, dv_ttm DOUBLE, total_share DOUBLE, float_share DOUBLE, free_share DOUBLE, total_mv DOUBLE, circ_mv DOUBLE, PRIMARY KEY (ts_code, trade_date))"

:: Alternative method: write fundamentals directly into Dolt via SQL server
::start "dolt-sql-server" dolt sql-server
::timeout /t 5 /nobreak >nul
::"%PYTHON%" "%INVESTMENT_DATA_DIR%tushare\update_a_stock_fundamental.py"
::taskkill /IM dolt.exe /F >nul 2>&1

::"%PYTHON%" "%INVESTMENT_DATA_DIR%tushare\dump_a_stock_fundamental.py" --start_date=%fund_startdate%
for %%F in ("%INVESTMENT_DATA_DIR%tushare\astock_fundamental\*.*") do (
  dolt table import -u -pk ts_code,trade_date ts_a_stock_fundamental "%%~fF"
)

dolt sql --file "%INVESTMENT_DATA_DIR%tushare\regular_update.sql"

dolt add -A

:: Determine if there are changes to commit
dolt status | findstr /C:"nothing to commit, working tree clean" >nul
if %ERRORLEVEL%==0 (
  echo No changes to commit. Working tree is clean.
) else (
  echo Changes found. Committing and pushing...
  ::dolt commit -m "Daily update"
  ::dolt push
  echo Changes committed and pushed.
)

popd
endlocal


