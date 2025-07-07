# whisper-web

### Screen Layout

1. **Main Transcription Area**: displayed on the left (full width when LLM correction is off)  
2. **LLM Correction Area**: displayed on the right (only when LLM correction is on)  
3. **Summary Display Area**: displayed at the bottom (after summary is executed)  

### LLM Typo Correction Feature

- Enable by checking “LLM Typo Correction”  
- Automatically correct the latest 10 transcriptions every 5 minutes  
- Display corrections in the dedicated right-hand area  
- Show a loading icon while corrections are in progress  

### About WhisperX

WhisperX 3.4.2 is a high-precision speech recognition model that extends OpenAI Whisper:  
- **High-precision recognition**: word-level timestamps and alignment  
- **Speaker separation**: identifies multiple speakers (coming soon)  
- **Fast processing**: efficient inference via batching  
- **Multilingual support**: supports multiple languages, including Japanese  

> **Note:** WhisperX requires additional model downloads, so the first startup may take some time.

# Whisper Real-time Transcription System

A system that captures browser-tab audio (e.g. Webex) in real time and displays it as captions.

## Features

- Capture tab audio using the browser’s `getDisplayMedia` API  
- High-precision recognition with the Whisper large-v3 model  
- **Three backends available**:  
  - faster-whisper 1.1.1 (fast processing)  
  - OpenAI Whisper v20250626 (latest)  
  - WhisperX 3.4.2 (high precision & speaker separation)  
- Real-time streaming via WebSocket  
- 10-second chunk processing for improved accuracy  
- Captions displayed with ~10 s latency  
- **Persistent transcription history** (text remains after stopping)  
- Supports multiple simultaneous clients  
- **Simple UI design**  
- **Light/Dark mode toggle**  
- **Language selection** (Auto-detect / Japanese / English)  
- **Copy transcription** feature  
- **Clear transcription** feature  
- **Timestamp display** feature  
- **Advanced parameter settings** (prompt, beam size, chunk size, etc.)  
- **LLM-based typo correction** (auto every 5 minutes)
- **LLM-based summarization**
- **Bottom summary display area**
- **Model sharing** for efficient GPU memory usage
- **Real-time GPU VRAM usage monitoring**

## Requirements

- Docker & Docker Compose  
- NVIDIA GPU (CUDA 12.x support)  
- NVIDIA Container Toolkit  
- ≥ 8 GB GPU memory (for large-v3 model)  
- Additional ~2 GB GPU memory for WhisperX  

## Setup & Startup

1. Clone the repo and navigate into the project directory:  
   ```bash
   cd whisper
   ```
2. Configure environment variables:

   ```bash
   cp .env.example .env
   # Edit .env to set your LLM API key, etc.
   ```
3. Start with Docker Compose:

   ```bash
   docker compose up --build
   ```

   > First run may take time to download Whisper models (\~3 GB).
4. Open your browser at `http://localhost:5173`
5. Click **Start Caption**
6. Select the tab to share (e.g. Webex) and check **Share audio**
7. When audio plays, captions will appear after a few seconds

## Usage

### Basic Controls

* **Start Caption**: begin audio capture & transcription
* **Stop Caption**: stop transcription (history remains)
* **Advanced Parameters**: show/hide Whisper settings
* **Clear**: clear transcription history

### Caption Display

* Captions appear at the bottom with a semi-transparent black background
* Final text (`is_final=true`) appears in white
* Interim text is semi-transparent and updates on confirmation
* **Transcription history is persisted**

### Additional Features

* **Engine selection**: Faster Whisper / OpenAI Whisper / WhisperX
* **Language selection**: Auto-detect / Japanese / English
* **Light/Dark mode**: toggle with the 🌞/🌙 button
* **Copy**: copy full transcription to clipboard
* **Summarize**: generate a summary via LLM (requires API key)
* **Timestamps**: toggle display on/off
* **LLM Typo Correction**: enable for auto corrections every 5 minutes
* **Advanced parameters**:

  * **Prompt**: initial prompt for Whisper
  * **Beam Size**: beam search width (1–10)
  * **Chunk Duration**: chunk length (5–30 s)
  * **Temperature**: sampling temperature (0.0–1.0)
  * **VAD Filter**: voice activity detection on/off
  * Other settings

## Architecture

### Model Sharing System

The application uses a singleton model manager to share Whisper models across multiple connections:
- **Efficient memory usage**: Only one model instance per backend type
- **Automatic cleanup**: Idle models are released after 5 minutes
- **Active user tracking**: Monitor how many users are using each model
- **Dynamic loading**: Models are loaded on-demand when first needed

### Frontend

* React 18 + TypeScript + Vite
* Tailwind CSS for styling
* WebSocket for real-time communication
* Web Audio API for PCM audio processing

### Backend

* FastAPI (Python 3.11)
* **Two engines**:

  * faster-whisper 1.1.1 (large-v3)
  * OpenAI Whisper v20250626 (large-v3)
* 10 s chunked PCM processing
* Inference on CUDA-enabled GPU

## Configuration

### Environment Variables (`.env`)

#### Whisper Settings

* `WHISPER_MODEL`: model (tiny, base, small, medium, large, large-v2, large-v3)
* `WHISPER_DEVICE`: cuda/cpu
* `WHISPER_COMPUTE_TYPE`: float16/int8

#### LLM Settings (Summarization)

* `LLM_API_KEY`: OpenAI or compatible key
* `LLM_MODEL`: e.g. gpt-4, gpt-3.5-turbo
* `LLM_BASE_URL`: API endpoint

#### Whisper Parameters (Duplicate Suppression)

* `WHISPER_BEAM_SIZE`: beam count (default: 5)
* `WHISPER_TEMPERATURE`: sampling temperature (default: 0.0)
* `WHISPER_VAD_FILTER`: always true
* `WHISPER_VAD_MIN_SILENCE_DURATION_MS`: default 1000 ms
* `WHISPER_VAD_MAX_SPEECH_DURATION_S`: default 30 s
* `WHISPER_REPETITION_PENALTY`: default 1.1
* `WHISPER_NO_SPEECH_THRESHOLD`: default 0.6
* `WHISPER_CONDITION_ON_PREVIOUS_TEXT`: false (duplicate suppression)

> See `.env.example` for full details.

### Duplicate Suppression Mechanism

1. **10 s chunk processing**: splits audio into optimal segments
2. **Enhanced VAD**: uses Silero VAD in faster-whisper
3. **Similarity detection**: edit-distance checks
4. **Silence detection**: clears buffer after 3 s of silence
5. **Overlap handling**: 1 s overlap for seamless transcription
6. **Context setting**: `condition_on_previous_text=false`

#### Frontend Env

* `VITE_WS_URL`: WebSocket URL (default: ws\://localhost:8000/ws/stt)
* `VITE_API_URL`: API URL (default: [http://localhost:8000](http://localhost:8000))

#### Backend Env

* `TORCH_CUDA_ARCH_LIST`: CUDA arch list (default: 8.0)
* `CUDA_VISIBLE_DEVICES`: GPU index (default: 0)

## Troubleshooting

### Audio Capture Issues

* Use Chrome/Edge (Firefox is limited)
* Tab audio not supported in macOS Safari
* Ensure **Share audio** is checked

### No Captions

* Check browser console for errors
* Verify WebSocket connection
* Inspect backend logs: `docker compose logs backend`

### GPU Errors

* Confirm NVIDIA Container Toolkit is installed
* Run `nvidia-smi` to verify GPU detection
* Check Docker Compose reservations settings

## Known Limitations

* macOS/Safari restricts tab audio sharing
* Some tabs (DRM-protected) may not capture audio
* Initial model loading \~10–20 s
* High concurrency may increase latency

## License
Released under the MIT License.
See each library’s documentation for individual licenses.