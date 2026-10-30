@echo off
REM =====================================================================
REM  ONE-TIME SETUP - run this first on a new PC (just double-click it).
REM  Creates the .venv environment and installs the needed libraries.
REM  Requires Python 3.10, 3.11, or 3.12 installed.
REM  (MediaPipe does NOT support Python 3.13 yet.)
REM =====================================================================
cd /d "%~dp0"

echo Looking for a compatible Python (3.10 - 3.12)...
set PYEXE=
for %%V in (3.12 3.11 3.10) do (
  if not defined PYEXE (
    py -%%V --version >nul 2>&1 && set PYEXE=py -%%V
  )
)
if not defined PYEXE (
  where python >nul 2>&1 && set PYEXE=python
)
if not defined PYEXE (
  echo.
  echo Could not find Python. Install Python 3.12 from:
  echo   https://www.python.org/downloads/release/python-3120/
  echo During install, tick "Add python.exe to PATH", then run setup.bat again.
  pause
  exit /b 1
)

echo Using: %PYEXE%
%PYEXE% -c "import sys; v=sys.version_info; exit(0 if (v.major==3 and 10<=v.minor<=12) else 1)"
if errorlevel 1 (
  echo.
  echo WARNING: the Python found is not 3.10-3.12. MediaPipe will likely fail.
  echo Install Python 3.12 from https://www.python.org/downloads/ and run again.
  pause
)

echo Creating virtual environment (.venv)...
%PYEXE% -m venv .venv
if errorlevel 1 ( echo Failed to create the venv. & pause & exit /b 1 )

echo Installing libraries - this can take a few minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 ( echo Install failed. Check the messages above. & pause & exit /b 1 )

echo.
echo ============================================================
echo  Setup complete!  Next: double-click  run_3_detect.bat
echo  (a trained model is already included), or run_1_collect.bat
echo  to record your own hands first.
echo ============================================================
pause
