@echo off
echo 🎬 Movie Recommendation System Setup
echo ====================================

echo.
echo 📋 Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found!
    echo.
    echo 🔧 Please install Python first:
    echo    1. Go to https://www.python.org/downloads/
    echo    2. Download Python 3.11 or 3.12
    echo    3. Run installer and CHECK "Add Python to PATH"
    echo    4. Restart this script
    echo.
    pause
    exit /b 1
)

echo ✅ Python found!
python --version

echo.
echo 📦 Installing dependencies...
pip install -r requirements.txt

echo.
echo 🚀 Starting the recommendation system...
echo.
echo 🌐 Web Interface: http://localhost:5000
echo 📊 API Endpoints: http://localhost:5000/api/
echo.
echo Press Ctrl+C to stop the server
echo.

python app_with_training.py

pause
