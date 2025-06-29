@echo off
REM Photography Automation Launcher Script

cd /d "/Users/gideoncrawley/Business Projects/InstaToEtsy"
call photography_env\Scripts\activate.bat

echo 🎨 Instagram to Etsy Photography Automation
echo ==========================================

REM Check if .env file exists
if not exist .env (
    echo ❌ .env file not found. Please copy .env.example to .env and configure your credentials.
    pause
    exit /b 1
)

REM Get profile URL from user if not provided
set PROFILE=%1
if "%PROFILE%"=="" (
    echo 📱 Enter your Instagram profile URL:
    set /p PROFILE=
)

REM Set defaults
if "%2"=="" (set CATEGORIES=sunset landscape mountains) else (set CATEGORIES=%2)
if "%3"=="" (set MAX_PHOTOS=10) else (set MAX_PHOTOS=%3)

echo 🚀 Starting automation with:
echo    Profile: %PROFILE%
echo    Categories: %CATEGORIES%
echo    Max photos: %MAX_PHOTOS%
echo.

REM Run the automation
python -m src.enhanced_main ^
    --profile "%PROFILE%" ^
    --categories %CATEGORIES% ^
    --max-photos %MAX_PHOTOS%

echo ✅ Automation completed! Check data\ folder for results.
pause
