# Project Plan: Local AI Video Clipper (Desktop)

## 📌 Project Overview
An open-source, fully local alternative to cloud-based video clipping platforms (inspired by Opus Clip and SupoClip). This tool processes long-form horizontal video (16:9) entirely on a local machine and outputs high-engagement vertical short clips (9:16) using an Electron frontend and a Python backend architecture.

### Target Hardware Environment
*   **Host OS for Initial Development:** Windows 11 (Discrete GPU Focus)
*   **Cross-Compatibility Targets:** Linux (Debian-based stable, Arch-based CachyOS)
*   **System Specs:** AMD Ryzen 7 7435 (8C/16T), 16GB DDR4 RAM, NVIDIA RTX 4050 Laptop GPU (6GB VRAM)
*   **Hardware Constraint Strategy:** Mandatory INT8 Quantization (`compute_type="int8"`) and `large-v3-turbo` model weights across deep learning engines to protect the 6GB VRAM layout from Out-Of-Memory (OOM) faults.

---

## 🛠️ Core Tech Stack
*   **Frontend UI:** Electron + React (TypeScript)
*   **Local Server Integration:** FastAPI (Python 3.11+)
*   **Video Orchestration:** FFmpeg static binaries (`ffmpeg`, `ffprobe`)
*   **Deep Learning Frameworks:** `faster-whisper`, `mediapipe` (or `ultralytics` YOLOv8-pose)

---

## 🚀 Feature Pipelines & Implementation Specifications

### 1. Long to Shorts (Local Engagement Engine)
*   **Objective:** Automatically cut a long video into short, engaging narratives without cloud token costs.
*   **Implementation Steps:**
    1. Parse the text output from the `faster-whisper` JSON module containing word-level timestamps.
    2. Feed the raw chunked text to a local LLM environment (via `Ollama` running Llama 3 / Mistral) or a high-efficiency multimodal API.
    3. **System Prompt for Agent Execution:**
       > *"You are an expert video editor. Analyze this timestamped transcript and return a JSON array containing the start and end timestamps of the 3 most engaging, cohesive, and standalone stories."*
    4. Capture JSON timestamps (`start_time`, `end_time`) and invoke FFmpeg via sub-process command execution utilizing the raw stream copy flag (`-c copy`) to extract the visual fragments instantaneously without re-encoding the entire structure.

### 2. AI Captions (Local Transcription & Rendering)
*   **Objective:** Generate dynamic, kinetic typography stylized for mobile viewing.
*   **Implementation Steps:**
    1. Isolate the audio layer into a lightweight `.wav` format utilizing an optimized FFmpeg filter chain.
    2. Instantiate `faster-whisper` pointing to the `large-v3-turbo` model weight directory. Configure the model instantiation string precisely as:
       ```python
       from faster_whisper import WhisperModel
       model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8")
       ```
    3. Enable `word_timestamps=True` during processing inference loops.
    4. Group sequential token responses into tight textual segments (maximum 2-3 words per frame context).
    5. Dynamically write out an **Advanced SubStation Alpha (.ass)** file mapping structural elements like primary/secondary fonts, hex colors (e.g., neon highlights), spatial margins, and physics-based rendering entry vectors (pop/bounce animations).
    6. Run an FFmpeg rendering pipeline to hardcode the ASS layout elements into the source video frame canvas.

### 3. AI Sound Effect (Contextual Trigger Engine)
*   **Objective:** Inject contextual, impactful sound effects automatically based on semantic and structural actions.
*   **Implementation Steps:**
    1. **Text Triggering:** Run a regex (`re`) engine across the transcription strings to match specific tokens representing sudden turns or high energy (e.g., *"boom"*, *"stop"*, *"look"*).
    2. **Visual Triggering:** Implement a structural scene transition detector via `OpenCV` tracking frame-by-frame structural similarity (SSIM metrics). A significant variance floor identifies a natural visual cut.
    3. **Audio Injection:** Map the target timestamps and map custom assets from a local catalog directory of audio flags (whooshes, clicks, ambient impacts). Combine the assets onto the main multi-track stream by calling an FFmpeg `amix` audio mixing filter or deploying specialized frame operations via `MoviePy`. Ensure automatic ducking routines are active so key voice dialogues remain loud and distinct.

### 4. AI Reframe (Dynamic Region of Interest Tracking)
*   **Objective:** Translate horizontal 16:9 widescreen captures into clear vertical 9:16 assets without fixed static letterboxing.
*   **Implementation Steps:**
    1. Deploy `OpenCV` to stream individual frames as array data objects to a local target.
    2. Pass the sequence data layers directly through a lightweight `MediaPipe Face Detection` or `YOLOv8-pose` inference loop to map absolute face tracking coordinates.
    3. Construct a standard moving-average smoothing calculation vector spanning an operational window of 10–15 frame ticks. This calculation normalizes rapid or jerky small movements of the subject's face.
    4. Translate the smoothed boundary coordinates into a mathematically accurate target 9:16 bounding box region.
    5. Instruct the rendering backend to crop out the calculated target 9:16 coordinate map using either standard array index operations or specialized FFmpeg crop filter parameters (`-vf crop=w:h:x:y`).

---

## 📅 Phased Execution Plan (Windows Host Focus)

### 🛠️ Phase 1: Pure Python Terminal Implementation
*Goal: Get the underlying core engine functioning end-to-end via CLI scripts.*

#### Step 1: Initialize Virtual Environment & CUDA
```bash
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
# Install CUDA-enabled PyTorch directly for the RTX 4050
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
# Install processing libraries
pip install faster-whisper mediapipe opencv-python fastapi uvicorn