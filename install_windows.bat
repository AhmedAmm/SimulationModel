@echo off
echo 🚀 Real-time Presentation System - Windows Installation
echo =========================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Upgrade pip
echo 🔧 Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch first (CPU version for compatibility)
echo 🔧 Installing PyTorch (CPU version)...
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Install other core dependencies
echo 🔧 Installing core dependencies...
python -m pip install numpy scipy requests

REM Try to install PyAudio
echo 🔧 Installing PyAudio (this might take a moment)...
python -m pip install pyaudio
if errorlevel 1 (
    echo ⚠️ PyAudio installation failed, trying alternative method...
    python -m pip install pipwin
    python -m pipwin install pyaudio
    if errorlevel 1 (
        echo ❌ PyAudio installation failed
        echo 📝 Please install manually:
        echo    1. Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
        echo    2. pip install downloaded_wheel.whl
        pause
        exit /b 1
    )
)

REM Install Whisper
echo 🔧 Installing OpenAI Whisper...
python -m pip install openai-whisper

REM Test installations
echo 🔍 Testing installations...
python -c "import torch; print('✅ PyTorch:', torch.__version__)" || echo "❌ PyTorch failed"
python -c "import numpy; print('✅ NumPy:', numpy.__version__)" || echo "❌ NumPy failed"
python -c "import scipy; print('✅ SciPy:', scipy.__version__)" || echo "❌ SciPy failed"
python -c "import pyaudio; print('✅ PyAudio: OK')" || echo "❌ PyAudio failed"
python -c "import whisper; print('✅ Whisper: OK')" || echo "❌ Whisper failed"

echo.
echo 🎉 Installation completed!
echo.
echo 📋 Next steps:
echo    1. Install FFmpeg from https://ffmpeg.org/download.html
echo    2. Add FFmpeg to your system PATH
echo    3. Run: python test_whisper.py (to test)
echo    4. Run: python main.py (for full system)
echo.
pause