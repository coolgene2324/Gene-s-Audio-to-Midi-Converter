# Gene's Audio to MIDI Converter & YouTube Downloader

A versatile, locally-hosted web application that provides AI-powered piano transcription to convert audio files into MIDI, and a full-featured YouTube downloader for saving video and audio content.

![maybe screenshot here](.png) 

---

## Features

This tool combines two major functionalities into a single, easy-to-use interface:

### 🎹 **AI MIDI Converter**
- **YouTube to MIDI:** Paste one or more YouTube URLs to directly convert them into MIDI files.
- **Audio File to MIDI:** Upload your own audio files (`.mp3`, `.wav`, `.flac`, etc.) for transcription.
- **Batch Processing:** Process multiple URLs and files in a single session.
- **Advanced Trimming:** Specify start and end times to convert only a specific segment of an audio file.
- **Custom Naming:** Assign a custom base name to your output files.
- **GPU Acceleration:** Automatically uses a CUDA-enabled GPU if available for significantly faster transcription.

### ⬇️ **YouTube Downloader (IN DEVELOPMENT)**
- **Download Video & Audio:** Save content from YouTube as either video (`.mp4`) or audio-only (`.mp3`/`.m4a`).
- **Quality Selection:** Fetches all available quality options, allowing you to choose the perfect resolution or bitrate for your needs.
- **Detailed Format Info:** See file extensions, resolution, framerate, and estimated file size before you download.
- **Simple Interface:** Just paste a URL, fetch formats, and click download.

---

## Setup and Installation

To run this application on your local machine, you will need **Python 3.8+** and **FFmpeg**.

### Step 1: Install FFmpeg

FFmpeg is a required backend tool for audio processing.

-   **Windows:** Download a build from [ffmpeg.org](https://ffmpeg.org/download.html). Extract the files and add the `bin` folder to your system's PATH environment variable.
-   **macOS (Homebrew):** `brew install ffmpeg`
-   **Linux (apt):** `sudo apt update && sudo apt install ffmpeg`

### Step 2: Set Up the Project

1.  **Clone or Download:** Get the project files from this repository.
2.  **Navigate to Folder:** Open a terminal or command prompt and navigate into the project's root directory.
3.  **Run Setup Script (Windows):** If you are on Windows, simply double-click the `setup.bat` file. It will create the necessary folders and install all Python packages.
4.  **Manual Installation (macOS/Linux):**
    - Create the `templates` directory: `mkdir templates`
    - Install the required Python packages:
      ```bash
      pip install -r requirements.txt
      ```

*Note on PyTorch: The `requirements.txt` file installs the CPU version of PyTorch. If you have a CUDA-enabled NVIDIA GPU, you can achieve a significant speedup by installing the GPU version. See the [official PyTorch website](https://pytorch.org/get-started/locally/) for the correct command for your system.*

---

## How to Run

1.  Make sure you have completed the setup steps above.
2.  Open a terminal in the project's root directory.
3.  Run the Flask application:
    ```bash
    python app.py
    ```
4.  Open your web browser and navigate to: **http://127.0.0.1:5000**

The application interface should now be visible and ready to use.

---

## How to Use

The application is divided into two tabs:

### MIDI Converter
1.  Paste one or more YouTube URLs into the text area, or upload audio files from your computer.
2.  Adjust the settings (trim times, custom name, etc.) as needed.
3.  Click **"Start Conversion"**.
4.  Progress will be displayed in the status and log sections.
5.  When finished, download links for your MIDI files will appear.

### YouTube Downloader (notryet implemented)
1.  Paste a single YouTube URL into the input field.
2.  Click **"Fetch Available Formats"**.
3.  Wait for the quality options for both video and audio to appear.
4.  Select your desired quality from the dropdown menus.
5.  Click **"Download Video"** or **"Download Audio"**.
6.  The file will be downloaded and a link will appear in the progress section.

---

## Technologies Used

-   **Backend:** Python, Flask
-   **Frontend:** HTML5, CSS3, JavaScript
-   **AI Transcription:** `piano-transcription-inference` (PyTorch)
-   **YouTube Interaction:** `yt-dlp`
-   **Audio Processing:** `librosa`, `ffmpeg-python`
