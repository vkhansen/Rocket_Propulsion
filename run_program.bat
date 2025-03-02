@echo off
cd /d "%~dp0"
cd Stage_Opt

echo Cleaning up previous run files...

REM Clean logs directory
if exist "logs\*" (
    echo Deleting files in logs directory...
    del /Q "logs\*"
    echo Logs directory cleaned.
) else (
    echo No files found in logs directory.
)

REM Clean output directory
if exist "output\*" (
    echo Deleting files in output directory...
    del /Q "output\*"
    echo Output directory cleaned.
) else (
    echo No files found in output directory.
)

echo Cleanup complete.
echo.
echo Starting optimization...

REM Run the main program
python main.py input_data.json
