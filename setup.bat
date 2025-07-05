@echo off
ECHO ==========================================================
ECHO.
ECHO      MIDI Converter Project Setup Script for Windows
ECHO.
ECHO ==========================================================
ECHO.

REM --- Step 1: Create project directories ---
ECHO [1/4] Creating project directory structure...
IF NOT EXIST "templates" (
    mkdir templates
    ECHO      - 'templates' directory created.
) ELSE (
    ECHO      - 'templates' directory already exists.
)
ECHO.

REM --- Step 2: Create requirements.txt file ---
ECHO [2/4] Generating requirements.txt file...
(
    echo Flask
    echo ffmpeg-python
    echo piano-transcription-inference
    echo yt-dlp
    echo librosa
    echo torch
    echo pretty-midi
    echo unidecode
    echo numpy
) > requirements.txt
ECHO      - 'requirements.txt' has been created.
ECHO.

REM --- Step 3: Check for FFmpeg ---
ECHO [3/4] Checking for FFmpeg installation...
where ffmpeg >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    ECHO      - [WARNING] FFmpeg not found in your system's PATH.
    ECHO        FFmpeg is required for audio processing.
    ECHO.
    ECHO        Opening the download page for you now. Please download
    ECHO        it, extract it, and add its 'bin' folder to your
    ECHO        Windows Environment PATH.
    ECHO.
    start "" "https://ffmpeg.org/download.html"
) ELSE (
    ECHO      - FFmpeg found.
)
ECHO.

REM --- Step 4: Install Python packages ---
ECHO [4/4] Installing required Python packages via pip...
ECHO      This may take several minutes.
ECHO.
pip install -r requirements.txt

ECHO.
ECHO ==========================================================
ECHO.
ECHO      Setup Complete!
ECHO.
ECHO      Next steps:
ECHO      1. Ensure 'app.py' and 'templates/index.html' are saved.
ECHO      2. If you had to install FFmpeg, please restart this script
ECHO         or your terminal to ensure it's detected.
ECHO      3. Run the application with: python app.py
ECHO.
ECHO ==========================================================
ECHO.

REM Pause to keep the window open so the user can see the output
pause
