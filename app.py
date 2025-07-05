import sys
import subprocess
import importlib
import os
import re
import shutil
import tempfile
import time
import numpy as np
import json
import zipfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import webbrowser # Import the webbrowser module

# --- Package Installation Check ---
# It's recommended to install these via requirements.txt, but this check remains for convenience.
REQUIRED_PACKAGES: Dict[str, str] = {
    'flask': 'Flask',
    'ffmpeg': 'ffmpeg-python',
    'piano_transcription_inference': 'piano-transcription-inference',
    'yt_dlp': 'yt-dlp',
    'librosa': 'librosa',
    'torch': 'torch',
    'pretty_midi': 'pretty-midi',
    'unidecode': 'Unidecode',
}

def check_and_install_packages(packages: Dict[str, str]) -> None:
    """Check if required packages are installed and print instructions for missing ones."""
    missing_packages = []
    for module_name, package_name in packages.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_packages.append(package_name)
    if missing_packages:
        print("🔴 Error: Missing required packages.")
        print("Please install them by running the following command in your terminal:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)

check_and_install_packages(REQUIRED_PACKAGES)

# --- Main Imports ---
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import torch
import librosa
import ffmpeg
import yt_dlp
from piano_transcription_inference import PianoTranscription, sample_rate
import pretty_midi
from unidecode import unidecode

# --- Flask App Setup ---
app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"

# Create necessary directories
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Utility Functions (Global Scope) ---
def chunk_audio(audio: np.ndarray, sr: int, chunk_sec: float = 10.0) -> List[np.ndarray]:
    """Split audio into chunks."""
    if len(audio) == 0:
        return []
    chunk_size = int(chunk_sec * sr)
    if chunk_size <= 0:
        raise ValueError("Invalid chunk size")
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
    return [c for c in chunks if len(c) > sr * 0.1]

# --- Utility Functions for Server-Sent Events (SSE) ---
def sse_log(message: str, level: str = "info"):
    """Formats a log message for Server-Sent Events (SSE)."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    return f"event: log\ndata: {{\"level\": \"{level}\", \"message\": \"{message}\", \"timestamp\": \"{timestamp}\"}}\n\n"

def sse_status(message: str, operation: str):
    """Formats a status update for SSE."""
    return f"event: status\ndata: {{\"message\": \"{message}\", \"operation\": \"{operation}\"}}\n\n"

def sse_result(files: List[str]):
    """
    Formats a result message for SSE.
    For this application, the result event will trigger the "Open Output Folder" button.
    """
    # The actual file list is not needed on the frontend for this purpose,
    # but we keep the structure for consistency.
    files_json_str = json.dumps(files)
    return f"event: result\ndata: {{\"files\": {files_json_str}}}\n\n"

def sse_progress(percentage: float, message: str):
    """Formats a progress update for SSE."""
    return f"event: progress\ndata: {{\"percentage\": {percentage}, \"message\": \"{message}\"}}\n\n"

def sse_error_message(message: str):
    """Formats a user-friendly error message for SSE."""
    return f"event: error_message\ndata: {{\"message\": \"{message}\"}}\n\n"


# --- Rate Limiter ---
class RateLimiter:
    """Rate limiter for YouTube requests."""
    def __init__(self, min_interval: float = 2.0):
        self.min_interval = min_interval
        self.last_request = 0

    def wait_if_needed(self):
        current_time = time.time()
        elapsed = current_time - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()

# --- Converter Class ---
class YouTubeToMidiConverter:
    """Main converter class adapted for local use and SSE logging."""

    def __init__(self, chunk_sec: float = 10.0, event_stream_queue=None,
                 onset_threshold: float = 0.5, offset_threshold: float = 0.5,
                 velocity_threshold: float = 0.5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize transcriber. The thresholds are set as attributes after creation
        self.transcriber = PianoTranscription(device=self.device)
        
        # Set transcription parameters as attributes
        self.transcriber.onset_threshold = onset_threshold
        self.transcriber.offset_threshold = offset_threshold
        self.transcriber.velocity_threshold = velocity_threshold

        self.sample_rate = sample_rate
        self.chunk_sec = chunk_sec
        self.rate_limiter = RateLimiter()
        self.temp_files = [] # List to track temporary files for cleanup
        self.max_file_size_mb = 500
        self.max_duration_minutes = 60
        self.event_stream_queue = event_stream_queue

        self.log(f"Converter initialized (Device: {self.device}). Transcription parameters: Onset={onset_threshold}, Offset={offset_threshold}, Velocity={velocity_threshold}")
        # Send initial progress for model loading, which can take time
        self.event_stream_queue.put(sse_progress(5, "Loading AI model..."))


    def log(self, message: str, level: str = "info"):
        if self.event_stream_queue:
            self.event_stream_queue.put(sse_log(message, level))
            print(f"[LOG - {datetime.now().strftime('%H:%M:%S')}][{level.upper()}]: {message}")

    def update_status(self, message: str, operation: str):
        if self.event_stream_queue:
            self.event_stream_queue.put(sse_status(message, operation))
            print(f"[STATUS - {datetime.now().strftime('%H:%M:%S')}][{operation}]: {message}")

    @staticmethod
    def is_valid_youtube_url(url: str) -> bool:
        youtube_regex = (
            r'(https?://)?(www\.)?'
            r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
            r'(watch\?v=|embed/|v/|.+\?v=)?([^\s&]+)'
        )
        return re.match(youtube_regex, url) is not None

    @staticmethod
    def sanitize_filename(name: str, max_length: int = 200) -> str:
        name = unidecode(name)
        name = re.sub(r'[^\w\s-]', '_', name).strip()
        name = re.sub(r'[-\s]+', '-', name)
        if not name:
            name = 'untitled'
        return name[:max_length]

    def cleanup_temp_files(self):
        """Cleans up temporary files generated during the process."""
        cleaned = 0
        for file_path in self.temp_files:
            try:
                if Path(file_path).exists():
                    Path(file_path).unlink()
                    cleaned += 1
            except Exception as e:
                self.log(f"Could not clean temp file {file_path}: {e}", "warning")
        if cleaned > 0:
            self.log(f"Cleaned up {cleaned} temporary files.")
        self.temp_files.clear()

    def get_unique_filepath(self, filepath: Path) -> Path:
        """Returns a unique filepath by appending a counter if the file already exists."""
        if not filepath.exists():
            return filepath
        base, suffix, parent, counter = filepath.stem, filepath.suffix, filepath.parent, 1
        while True:
            new_filepath = parent / f"{base}_{counter}{suffix}"
            if not new_filepath.exists():
                return new_filepath
            counter += 1

    def validate_audio_file(self, file_path: Path) -> Tuple[bool, str]:
        """Validates audio file size and duration."""
        self.log(f"Validating audio file: {file_path.name}")
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return False, f"File too large: {file_size_mb:.1f}MB (Max: {self.max_file_size_mb}MB)."
            duration = librosa.get_duration(path=str(file_path))
            if duration > self.max_duration_minutes * 60:
                return False, f"Audio too long: {duration/60:.1f} minutes (Max: {self.max_duration_minutes} minutes)."
            self.log(f"Audio file '{file_path.name}' valid. Size: {file_size_mb:.1f}MB, Duration: {duration:.1f}s.")
            return True, "Valid"
        except Exception as e:
            self.log(f"Error validating audio file {file_path.name}: {e}", "error")
            return False, f"Invalid audio file or unable to read duration: {e}"

    def download_mp3_from_youtube(self, youtube_url: str, output_dir: Path, custom_name: Optional[str] = None, idx: Optional[int] = None, total_tasks: int = 1, cookiefile: Optional[str] = None) -> Path:
        """Downloads an MP3 from a YouTube URL."""
        self.log(f"Attempting to download YouTube URL: {youtube_url}")
        if not self.is_valid_youtube_url(youtube_url):
            raise ValueError(f"Invalid YouTube URL provided: {youtube_url}")

        self.rate_limiter.wait_if_needed()
        self.update_status(f"Downloading {youtube_url}", "Downloading")
        self.event_stream_queue.put(sse_progress(10, f"Downloading: {youtube_url}")) # Initial download progress

        output_template = str(output_dir / '%(title)s.%(ext)s')
        ydl_opts = {
            'format': 'bestaudio/best', 'outtmpl': output_template,
            'noplaylist': True, 'quiet': True, 'no_warnings': True, 'ignoreerrors': True,
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
            'progress_hooks': [lambda d: self._download_progress_hook(d, youtube_url)] # Add progress hook
        }
        if cookiefile: ydl_opts['cookiefile'] = cookiefile

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(youtube_url, download=True)
                if not info_dict:
                    raise ValueError("Download failed, YouTube-DLP returned no information.")
                
                # yt-dlp might download as .webm then convert to .mp3
                # We need to find the final mp3 file.
                downloaded_file = Path(ydl.prepare_filename(info_dict))
                # If postprocessor changed extension to .mp3, find that file.
                final_ext_file = downloaded_file.with_suffix('.mp3')
                if not final_ext_file.exists():
                     # Fallback to original downloaded file if not converted to mp3
                    original_mp3_file = downloaded_file
                else:
                    original_mp3_file = final_ext_file

                if not original_mp3_file.exists():
                    raise FileNotFoundError(f"Downloaded file not found at expected path: {original_mp3_file}")

            title = custom_name or info_dict.get('title', 'video')
            filename = f"{self.sanitize_filename(title)}_{idx+1}.mp3" if total_tasks > 1 and idx is not None else f"{self.sanitize_filename(title)}.mp3"
            final_filepath = self.get_unique_filepath(output_dir / filename)
            
            # Use rename for efficiency if possible, otherwise move
            try:
                original_mp3_file.rename(final_filepath)
            except OSError: # If rename fails across filesystems
                shutil.move(str(original_mp3_file), str(final_filepath))

            self.temp_files.append(final_filepath)
            self.log(f"Downloaded: {final_filepath.name}", "success")
            self.event_stream_queue.put(sse_progress(25, f"Downloaded: {final_filepath.name}")) # Update progress after download
            return final_filepath
        except Exception as e:
            self.log(f"Error during YouTube download for {youtube_url}: {e}", "error")
            raise # Re-raise the exception to be caught by the outer try-except

    def _download_progress_hook(self, d, url):
        """Internal hook for yt-dlp to report download progress."""
        if d['status'] == 'downloading':
            total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            downloaded_bytes = d.get('downloaded_bytes', 0)
            if total_bytes > 0:
                percentage = (downloaded_bytes / total_bytes) * 100
                # Scale progress for overall app progress (e.g., 10-25% for download phase)
                scaled_percentage = 10 + (percentage * 0.15) # From 10% to 25%
                self.event_stream_queue.put(sse_progress(scaled_percentage, f"Downloading: {url} ({d['_percent_str'].strip()})"))
        elif d['status'] == 'finished':
            self.event_stream_queue.put(sse_progress(25, f"Finished downloading: {url}"))


    def trim_audio(self, input_path: Path, output_path: Path, start_time: float, end_time: Optional[float] = None) -> None:
        """Trims an audio file using ffmpeg."""
        self.log(f"Trimming audio: {input_path.name} from {start_time}s to {end_time or 'end'}s")
        self.update_status(f"Trimming {input_path.name}", "Processing")
        try:
            stream = ffmpeg.input(str(input_path))
            filter_spec = f"atrim=start={start_time}"
            if end_time and end_time > 0:
                filter_spec += f":end={end_time}"
            trimmed = stream.audio.filter_multi_output(filter_spec).filter('asetpts', 'PTS-STARTPTS')
            ffmpeg.output(trimmed, str(output_path)).run(overwrite_output=True, quiet=True)
            self.temp_files.append(output_path)
            self.log(f"Trimmed audio created: {output_path.name}")
        except ffmpeg.Error as e:
            error_message = f"FFmpeg error during trimming of {input_path.name}: {e.stderr.decode()}"
            self.log(error_message, "error")
            raise Exception(error_message) # Re-raise as generic exception
        except Exception as e:
            self.log(f"An unexpected error occurred trimming audio {input_path.name}: {e}", "error")
            raise

    def safe_transcribe(self, audio_chunk: np.ndarray, output_path: str):
        """Transcribes an audio chunk, with fallback to CPU if CUDA runs out of memory."""
        self.log(f"Transcribing audio chunk to {os.path.basename(output_path)}...")
        try:
            self.transcriber.transcribe(audio_chunk, output_path)
            self.log(f"Successfully transcribed audio chunk to {os.path.basename(output_path)}")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and self.device == 'cuda':
                self.log("GPU memory issue, retrying on CPU for this chunk.", "warning")
                # Create a new CPU transcriber with the same parameters
                cpu_transcriber = PianoTranscription(
                    device='cpu',
                    onset_threshold=self.transcriber.onset_threshold,
                    offset_threshold=self.transcriber.offset_threshold,
                    velocity_threshold=self.transcriber.velocity_threshold
                )
                cpu_transcriber.transcribe(audio_chunk, output_path)
                self.log(f"Successfully transcribed chunk on CPU to {os.path.basename(output_path)}.")
            else:
                self.log(f"Error during transcription to {os.path.basename(output_path)}: {e}", "error")
                raise # Re-raise other runtime errors

    def chunked_convert_audio_to_midi(self, audio_file_path: Path, midi_file_path: Path, max_duration: float = 2.0) -> bool:
        """Converts an audio file to MIDI in chunks, providing progress updates."""
        self.update_status(audio_file_path.name, "Transcribing")
        self.log(f"Starting chunked conversion for {audio_file_path.name}")
        
        try:
            duration = librosa.get_duration(path=str(audio_file_path))
            if duration == 0:
                self.log(f"Audio file {audio_file_path.name} has zero duration, skipping conversion.", "warning")
                self.event_stream_queue.put(sse_error_message(f"Audio file '{audio_file_path.name}' has no audible content or zero duration. Cannot transcribe."))
                return False

            total_chunks = int(np.ceil(duration / self.chunk_sec))
            self.log(f"Processing {audio_file_path.name} ({duration:.1f}s) in {total_chunks} chunks of {self.chunk_sec}s.")

            audio, _ = librosa.load(str(audio_file_path), sr=self.sample_rate, mono=True)
            # Call chunk_audio directly from the global scope (this is the key fix)
            audio_chunks = chunk_audio(audio, self.sample_rate, self.chunk_sec)
            merged_midi = pretty_midi.PrettyMIDI()
            current_time_offset = 0.0

            # Base progress for transcription phase (e.g., 25% to 95%)
            transcription_start_percentage = 25
            transcription_end_percentage = 95
            progress_range = transcription_end_percentage - transcription_start_percentage

            for i, chunk in enumerate(audio_chunks):
                self.log(f"Processing chunk {i+1}/{total_chunks}...")
                
                # Calculate and send chunk-specific progress
                chunk_progress = (i / total_chunks) * progress_range
                overall_progress = transcription_start_percentage + chunk_progress
                self.event_stream_queue.put(sse_progress(overall_progress, f"Transcribing chunk {i+1}/{total_chunks}"))

                with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
                    tmp_midi_path = Path(tmp.name)
                
                # Check if the chunk is not empty before transcribing
                if chunk.size == 0:
                    self.log(f"Skipping empty chunk {i+1}/{total_chunks}.", "warning")
                    tmp_midi_path.unlink(missing_ok=True) # Ensure empty temp file is removed
                    current_time_offset += len(chunk) / self.sample_rate # Still advance offset
                    continue

                self.safe_transcribe(chunk, str(tmp_midi_path))
                
                try:
                    # Check if the temporary MIDI file was actually created and has content
                    if not tmp_midi_path.exists() or tmp_midi_path.stat().st_size == 0:
                        self.log(f"Temporary MIDI file for chunk {i+1} was not created or is empty. Skipping merge.", "warning")
                        tmp_midi_path.unlink(missing_ok=True)
                        current_time_offset += len(chunk) / self.sample_rate
                        continue # Skip this chunk if MIDI is not valid

                    pm_chunk = pretty_midi.PrettyMIDI(str(tmp_midi_path))
                except Exception as midi_read_err:
                    self.log(f"Error reading MIDI from temporary file {tmp_midi_path.name} for chunk {i+1}: {midi_read_err}", "error")
                    tmp_midi_path.unlink(missing_ok=True)
                    continue # Skip this chunk if MIDI is unreadable

                tmp_midi_path.unlink(missing_ok=True) # Clean up temp MIDI file

                for instr in pm_chunk.instruments:
                    for note in instr.notes:
                        note.start += current_time_offset
                        note.end += current_time_offset
                        if (note.end - note.start) > max_duration:
                            note.end = note.start + max_duration
                    merged_midi.instruments.append(instr)
                current_time_offset += len(chunk) / self.sample_rate

            merged_midi.write(str(midi_file_path))
            self.log(f"Final MIDI created: {midi_file_path.name}", "success")
            self.event_stream_queue.put(sse_progress(100, "MIDI conversion complete!")) # Final progress
            return True
        except Exception as e:
            self.log(f"Error during chunked conversion for {audio_file_path.name}: {e}", "error")
            self.event_stream_queue.put(sse_error_message(f"Conversion failed for '{audio_file_path.name}'. Details in log."))
            return False

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/output/<filename>')
def download_file(filename):
    """Serves files from the output directory for downloading (though replaced by open folder)."""
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

@app.route('/open_output_folder', methods=['POST'])
def open_output_folder():
    """Opens the output directory on the user's operating system."""
    try:
        folder_path = str(OUTPUT_DIR.resolve()) # Get absolute path
        print(f"Attempting to open folder: {folder_path}")
        if sys.platform == "win32":
            os.startfile(folder_path)
        elif sys.platform == "darwin": # macOS
            subprocess.Popen(["open", folder_path])
        else: # linux variants
            subprocess.Popen(["xdg-open", folder_path])
        return jsonify({"status": "success", "message": f"Opened folder: {folder_path}"})
    except Exception as e:
        print(f"[ERROR]: Failed to open output folder: {e}")
        return jsonify({"status": "error", "message": f"Failed to open folder: {e}. Check if the folder exists or permissions."}), 500


@app.route('/convert', methods=['POST'])
def convert():
    """Handles the conversion request and starts the SSE stream."""
    from queue import Queue
    event_queue = Queue()

    # Extract all necessary data from request before spawning thread
    form_data_dict = request.form.to_dict()
    uploaded_files_list = []
    
    # Process uploaded audio files
    for file_storage in request.files.getlist('audio_files'):
        if file_storage.filename:
            file_path = UPLOAD_DIR / file_storage.filename
            try:
                file_storage.save(file_path)
                uploaded_files_list.append(file_path)
                print(f"[INFO]: Saved uploaded file: {file_path.name}")
            except Exception as e:
                # Log to console and send an error to the frontend if file saving fails
                print(f"[ERROR]: Failed to save uploaded file {file_storage.filename}: {e}")
                event_queue.put(sse_error_message(f"Failed to save uploaded file '{file_storage.filename}'. Please check permissions."))
                event_queue.put(sse_log(f"Failed to save uploaded file {file_storage.filename}: {e}", "error"))

    # Process cookie file if present
    cookie_file_data = None
    cookie_file_storage = request.files.get('cookie_file')
    if cookie_file_storage and cookie_file_storage.filename:
        cookie_path = UPLOAD_DIR / cookie_file_storage.filename
        try:
            cookie_file_storage.save(cookie_path)
            cookie_file_data = str(cookie_path)
            print(f"[INFO]: Saved cookie file: {cookie_path.name}")
        except Exception as e:
            print(f"[ERROR]: Failed to save cookie file {cookie_file_storage.filename}: {e}")
            event_queue.put(sse_error_message(f"Failed to save cookie file '{cookie_file_storage.filename}'. Please check permissions."))
            event_queue.put(sse_log(f"Failed to save cookie file {cookie_file_storage.filename}: {e}", "error"))


    def event_stream():
        """Generator function that yields SSE messages."""
        while True:
            # Wait for a message from the worker thread
            message = event_queue.get()
            if message == "END":
                break
            yield message

    # Start the conversion process in a separate thread, passing extracted data
    threading.Thread(
        target=process_conversion_thread,
        args=(form_data_dict, uploaded_files_list, cookie_file_data, event_queue)
    ).start()

    # Return the streaming response
    return Response(event_stream(), content_type='text/event-stream')


def process_conversion_thread(form_data: Dict, uploaded_files_paths: List[Path], cookie_file_path: Optional[str], queue: 'Queue'):
    """The main processing logic, run in a background thread."""
    print("[THREAD START]: Conversion thread initiated.") # Console log for thread start
    converter = None # Initialize converter to None to prevent UnboundLocalError in except/finally
    try:
        # Get settings from form_data (which is now a regular dictionary)
        youtube_urls = form_data.get('youtube_urls', '').strip()
        custom_name = form_data.get('custom_name', '').strip() or None
        max_duration = float(form_data.get('max_duration', 0.2))
        start_time = float(form_data.get('start_time', 0.0))
        end_time = float(form_data.get('end_time', 0.0)) or None
        chunk_duration = float(form_data.get('chunk_duration', 10.0))
        zip_output = form_data.get('zip_files') == 'on'
        keep_mp3 = form_data.get('keep_mp3') == 'on'

        # Get transcription parameters
        onset_threshold = float(form_data.get('onset_threshold', 0.5))
        offset_threshold = float(form_data.get('offset_threshold', 0.5))
        velocity_threshold = float(form_data.get('velocity_threshold', 0.5))

        # Initialize converter with the queue for logging and new parameters
        converter = YouTubeToMidiConverter(
            chunk_sec=chunk_duration,
            event_stream_queue=queue,
            onset_threshold=onset_threshold,
            offset_threshold=offset_threshold,
            velocity_threshold=velocity_threshold
        )
        converter.log("Received conversion request parameters and initialized converter.")
        queue.put(sse_progress(10, "Parameters received, starting processing..."))


        # Handle cookie file data (already saved and path passed)
        if cookie_file_path:
            converter.log("Cookie file path received and processed.")
        else:
            converter.log("No cookie file provided.", "info")

        # uploaded_files_paths is already a list of Paths to saved files
        local_files_to_process = uploaded_files_paths

        # Re-check for empty inputs after processing file uploads/downloads in the main thread
        youtube_urls_list = [url.strip() for url in youtube_urls.splitlines() if url.strip()]
        total_tasks = len(youtube_urls_list) + len(local_files_to_process)
        if total_tasks == 0:
            converter.log("No files or URLs provided for conversion.", "error")
            queue.put(sse_error_message("No audio files or YouTube URLs were provided for conversion."))
            queue.put("END")
            return
            
        converter.log(f"Starting conversion for {total_tasks} item(s)...", "info")
        results = [] # To store paths of successfully generated MIDI files

        # Process YouTube URLs
        for idx, url in enumerate(youtube_urls_list):
            converter.log(f"Processing YouTube URL {idx+1}/{len(youtube_urls_list)}: {url}")
            try:
                mp3_path = converter.download_mp3_from_youtube(url, UPLOAD_DIR, custom_name, idx, total_tasks, cookiefile=cookie_file_path)
                
                valid, msg = converter.validate_audio_file(mp3_path)
                if not valid:
                    converter.log(f"Validation failed for {mp3_path.name}: {msg}", "error")
                    queue.put(sse_error_message(f"Skipping '{mp3_path.name}' due to validation error: {msg}"))
                    if mp3_path.exists(): mp3_path.unlink(missing_ok=True) # Clean up invalid downloaded file
                    continue

                audio_to_process = mp3_path
                current_trimmed_path = None # To keep track of the trimmed path for cleanup
                if start_time > 0 or (end_time is not None and end_time > 0):
                    current_trimmed_path = UPLOAD_DIR / f"trimmed_{mp3_path.stem}.mp3"
                    converter.trim_audio(mp3_path, current_trimmed_path, start_time, end_time)
                    audio_to_process = current_trimmed_path

                midi_file_name_stem = f"{converter.sanitize_filename(custom_name)}" if custom_name else converter.sanitize_filename(audio_to_process.stem)
                midi_file = OUTPUT_DIR / f"{midi_file_name_stem}.mid"
                
                success = converter.chunked_convert_audio_to_midi(audio_to_process, midi_file, max_duration)
                if success:
                    results.append(midi_file)
                else:
                    converter.log(f"Conversion failed for {audio_to_process.name}.", "error")

                # Clean up intermediate files if not keeping MP3s
                if not keep_mp3 and mp3_path.exists():
                    converter.log(f"Deleting downloaded MP3: {mp3_path.name}", "info")
                    mp3_path.unlink(missing_ok=True)
                if current_trimmed_path and current_trimmed_path.exists():
                    converter.log(f"Deleting trimmed audio: {current_trimmed_path.name}", "info")
                    current_trimmed_path.unlink(missing_ok=True)

            except Exception as e:
                converter.log(f"Failed to process YouTube URL {url}: {e}", "error")
                queue.put(sse_error_message(f"Failed to convert YouTube URL '{url}'. Check logs for details."))
                converter.cleanup_temp_files() # Ensure temporary files are cleaned up in case of error


        # Process uploaded files
        for audio_path in local_files_to_process:
            converter.log(f"Processing uploaded file: {audio_path.name}")
            try:
                valid, msg = converter.validate_audio_file(audio_path)
                if not valid:
                    converter.log(f"Validation failed for {audio_path.name}: {msg}", "error")
                    queue.put(sse_error_message(f"Skipping '{audio_path.name}' due to validation error: {msg}"))
                    if audio_path.exists(): audio_path.unlink(missing_ok=True) # Clean up invalid uploaded file
                    continue
                
                audio_to_process = audio_path
                current_trimmed_path = None # To keep track of the trimmed path for cleanup
                if start_time > 0 or (end_time is not None and end_time > 0):
                    current_trimmed_path = UPLOAD_DIR / f"trimmed_{audio_path.stem}.mp3"
                    converter.trim_audio(audio_path, current_trimmed_path, start_time, end_time)
                    audio_to_process = current_trimmed_path

                midi_file_name_stem = f"{converter.sanitize_filename(custom_name)}" if custom_name else converter.sanitize_filename(audio_to_process.stem)
                midi_file = OUTPUT_DIR / f"{midi_file_name_stem}.mid"
                
                success = converter.chunked_convert_audio_to_midi(audio_to_process, midi_file, max_duration)
                if success:
                    results.append(midi_file)
                else:
                    converter.log(f"Conversion failed for {audio_to_process.name}.", "error")

                # Clean up intermediate files
                if not keep_mp3 and audio_path.exists():
                     converter.log(f"Deleting uploaded MP3: {audio_path.name}", "info")
                     audio_path.unlink(missing_ok=True)
                if current_trimmed_path and current_trimmed_path.exists():
                    converter.log(f"Deleting trimmed audio: {current_trimmed_path.name}", "info")
                    current_trimmed_path.unlink(missing_ok=True)
            except Exception as e:
                converter.log(f"Failed to process file {audio_path.name}: {e}", "error")
                queue.put(sse_error_message(f"Failed to convert file '{audio_path.name}'. Check logs for details."))
                converter.cleanup_temp_files() # Ensure temporary files are cleaned up in case of error
        
        # Finalize and send results
        if not results:
            if converter:
                converter.log("No files were converted successfully.", "warning")
                converter.update_status("Conversion failed.", "Error")
            else:
                print("[ERROR]: No files converted, and converter object was not initialized.")
            queue.put(sse_log("No files were converted successfully. Check server logs for details.", "error"))
            queue.put(sse_status("Conversion failed.", "Error"))
            queue.put(sse_progress(0, "Conversion Failed")) # Reset progress on failure
        else:
            final_file_names = [f.name for f in results]
            if zip_output and len(results) > 1:
                zip_filename_obj = OUTPUT_DIR / f"midi_files_{int(time.time())}.zip"
                with zipfile.ZipFile(zip_filename_obj, 'w') as zipf:
                    for f in results:
                        zipf.write(f, arcname=f.name)
                final_file_names = [zip_filename_obj.name] # Only the zip file for download
                if converter:
                    converter.log(f"Created ZIP archive: {zip_filename_obj.name}", "success")
            
            # Send the result signal to client (which now triggers open folder button)
            queue.put(sse_result(final_file_names)) 
            if converter:
                converter.log("Conversion complete. Output available in folder.", "info")

        if converter: # Ensure converter is available before final logs
            converter.log("All tasks completed.", "success")
            converter.update_status("Ready for next conversion.", "Completed")
            queue.put(sse_progress(100, "All tasks completed!"))
        else: # Fallback console log if converter isn't available
            print("[INFO]: Conversion process completed (or failed early) - converter object not fully initialized.")

    except Exception as e:
        # Log any unexpected error during the process
        error_msg = f"An unexpected error occurred in the conversion thread: {e}"
        if converter:
            converter.log(error_msg, "error")
        else:
            print(f"[CRITICAL ERROR - {datetime.now().strftime('%H:%M:%S')}]: {error_msg}")
        queue.put(sse_log(error_msg, "error"))
        queue.put(sse_error_message("An unhandled error occurred during conversion. Check server logs."))
        queue.put(sse_status("An error occurred.", "Error"))
        queue.put(sse_progress(0, "Error Occurred")) # Reset or indicate error progress
    finally:
        # Signal that the stream is finished
        queue.put("END")
        print("[THREAD END]: Conversion thread finished.") # Console log for thread end

if __name__ == '__main__':
    print("🌍 Starting local MIDI Converter server...")
    print("✅ Open http://127.0.0.1:5000 in your web browser.")
    webbrowser.open("http://127.0.0.1:5000") # Automatically open the browser
    app.run(debug=True, port=5000)
