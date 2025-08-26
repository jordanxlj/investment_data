@echo on
setlocal EnableExtensions EnableDelayedExpansion

:: Usage: dump_bin.bat [WORKING_DIR] [QLIB_REPO]
:: - WORKING_DIR: Base directory to store dolt clone, qlib repo and output (defaults to current directory)
:: - QLIB_REPO:  Git URL for qlib (defaults to https://github.com/microsoft/qlib.git)

:: Arguments and defaults
set "WORKING_DIR=%~1"
if "%WORKING_DIR%"=="" set "WORKING_DIR=%CD%"
:: for %%I in ("%WORKING_DIR%") do set "WORKING_DIR=%%~fI"

:: set "QLIB_REPO=%~2"
:: if "%QLIB_REPO%"=="" set "QLIB_REPO=https://github.com/microsoft/qlib.git"

:: Paths
set "PROJECT_DIR=%~dp0"
for %%I in ("%PROJECT_DIR%") do set "PROJECT_DIR=%%~fI"
set "PYTHON=%PYTHON%"
if "%PYTHON%"=="" set "PYTHON=python"

set "DOLT_BASE=%WORKING_DIR%\dolt"
set "DOLT_REPO_DIR=%DOLT_BASE%\investment_data"
set "QLIB_DIR=%WORKING_DIR%\qlib"
set "QLIB_BIN_DIR=%WORKING_DIR%\qlib_bin"
set "PYTHONPATH=.;%QLIB_DIR%;%QLIB_DIR%\scripts"

:: Pre-flight checks
where dolt >nul 2>&1
if errorlevel 1 (
  echo ERROR: dolt is not installed or not in PATH. Please install dolt first: https://github.com/dolthub/dolt
  exit /b 1
)

where "%PYTHON%" >nul 2>&1
if errorlevel 1 (
  echo ERROR: %PYTHON% not found in PATH. Please install Python and ensure it is available.
  exit /b 1
)

:: Ensure directories exist
if not exist "%DOLT_BASE%" mkdir "%DOLT_BASE%"
if not exist "%QLIB_BIN_DIR%" mkdir "%QLIB_BIN_DIR%"

:: Clone dolt repo if missing
if not exist "%DOLT_REPO_DIR%" (
  pushd "%DOLT_BASE%"
  dolt clone chenditc/investment_data
  if errorlevel 1 (
    echo Failed to clone dolt repo
    popd
    exit /b 1
  )
  popd
)

:: Clone qlib if missing
if not exist "%QLIB_DIR%" (
  git clone "%QLIB_REPO%" "%QLIB_DIR%"
  if errorlevel 1 (
    echo Failed to clone qlib repo from %QLIB_REPO%
    exit /b 1
  )
)

:: Start dolt SQL server
pushd "%DOLT_REPO_DIR%"
dolt pull origin master || exit /b 1
start "dolt-sql-server" dolt sql-server
popd

:: wait for sql server start
timeout /t 5 /nobreak >nul

:: Run conversion pipeline from project directory
pushd "%PROJECT_DIR%"
if not exist ".\qlib\qlib_source" mkdir ".\qlib\qlib_source"
"%PYTHON%" ".\qlib\dump_all_to_qlib_source.py"

set "PYTHONPATH=%PYTHONPATH%;%QLIB_DIR%\scripts"
"%PYTHON%" ".\qlib\normalize.py" normalize_data --source_dir ".\qlib\qlib_source" --normalize_dir ".\qlib\qlib_normalize" --max_workers=16 --date_field_name="tradedate"
"%PYTHON%" "%QLIB_DIR%\scripts\dump_bin.py" dump_all --csv_path ".\qlib\qlib_normalize" --qlib_dir "%QLIB_BIN_DIR%" --date_field_name=tradedate --exclude_fields=tradedate,symbol

if not exist ".\qlib\qlib_index" mkdir ".\qlib\qlib_index"
"%PYTHON%" ".\qlib\dump_index_weight.py"
"%PYTHON%" ".\tushare\dump_day_calendar.py" "%QLIB_BIN_DIR%"
popd

:: Stop dolt SQL server
taskkill /IM dolt.exe /F >nul 2>&1

:: Ensure instruments directory exists and copy index instruments
if not exist "%QLIB_BIN_DIR%\instruments" mkdir "%QLIB_BIN_DIR%\instruments"
copy /Y "%PROJECT_DIR%qlib\qlib_index\csi*" "%QLIB_BIN_DIR%\instruments\" >nul 2>&1

:: Package the result to a tar.gz in project directory
set "TARBALL=%PROJECT_DIR%qlib_bin.tar.gz"
if exist "%TARBALL%" del /f /q "%TARBALL%"
:: tar -czvf "%TARBALL%" -C "%WORKING_DIR%" qlib_bin

echo Generated tarball at "%TARBALL%"

:: Optionally move tarball to OUTPUT_DIR if provided and exists
if defined OUTPUT_DIR (
  if exist "%OUTPUT_DIR%" (
    move /Y "%TARBALL%" "%OUTPUT_DIR%\" >nul
    if not errorlevel 1 (
      echo Moved tarball to "%OUTPUT_DIR%\qlib_bin.tar.gz"
    ) else (
      echo Failed to move tarball to OUTPUT_DIR
    )
  ) else (
    echo OUTPUT_DIR is defined but does not exist: "%OUTPUT_DIR%"
  )
)

endlocal


