# LocalCut AI Clipper ✂️

An open-source, fully local alternative to cloud-based video clipping platforms (inspired by Opus Clip and SupoClip). 
This tool processes long-form horizontal video (16:9) entirely on your local machine and automatically outputs high-engagement vertical short clips (9:16) with burnt-in kinetic subtitles—without any cloud token costs or internet requirements!

## ✨ Features

- **Long-to-Shorts (Local Engagement Engine):** Automatically analyzes audio transcripts and uses local LLMs to find cohesive, standalone stories. Extracts these stories into separate, perfectly-timed video clips.
- **AI Reframe (Dynamic Face Tracking):** Uses MediaPipe and OpenCV to identify the prominent speaker. Employs a mathematically smoothed moving-average window to dynamically pan and crop the 16:9 widescreen video into a perfect 9:16 vertical layout without losing the subject.
- **AI Captions (Kinetic Typography):** Leverages `faster-whisper` for word-level timestamps to generate mobile-optimized dynamic subtitles (.ass format) with slight pop animations, burnt directly into the final video.
- **Privacy & Free:** Everything runs entirely on your local hardware. No API keys, no subscriptions, no data harvesting.

## 🛠️ Tech Stack

- **Core Engine:** Python 3.11+
- **Video Orchestration:** FFmpeg (Static Binaries)
- **Transcription Engine:** `faster-whisper` (large-v3-turbo model, INT8 Quantization for VRAM efficiency)
- **Computer Vision:** `OpenCV` and `MediaPipe` Face Detection
- **Narrative AI:** `Ollama` running local Llama 3 

## 🚀 How It Works

The pipeline is fully automated and runs sequentially:
1. **Transcription (`app/transcription.py`):** The audio stream is isolated and transcribed by Whisper, returning timestamped word data.
2. **Segmentation (`app/segmentation.py`):** The transcript is grouped into 8-minute blocks and sent to Ollama (Llama 3). The LLM isolates engaging narrative sections. Smart auto-padding ensures clips are a minimum of 45 seconds to preserve context.
3. **Reframing (`app/reframing.py`):** MediaPipe scans the cropped clips for human faces. It calculates a smoothed X-axis center point to dynamically slice out a mobile-friendly 9:16 segment from the video canvas. Audio is explicitly re-encoded to ensure compatibility.
4. **Subtitling (`app/subtitles.py`):** The grouped Whisper output is mapped to an Advanced SubStation Alpha (`.ass`) file to render kinetic, bold subtitles right in the center of the video, baked in instantaneously via FFmpeg.

## 💻 Setup & Installation

### Prerequisites
- Windows 10/11 (or Linux)
- A dedicated GPU is highly recommended (CUDA enabled)
- [FFmpeg](https://ffmpeg.org/download.html) installed and accessible in your system PATH.
- [Ollama](https://ollama.com/) installed with the `llama3` model downloaded.

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/LocalCut-AI-Clipper.git
   cd LocalCut-AI-Clipper
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install PyTorch with CUDA support:**
   *(Example for CUDA 12.1 - adjust based on your GPU requirements)*
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install Python dependencies:**
   ```bash
   pip install faster-whisper mediapipe opencv-python requests
   ```

5. **Start your local LLM:**
   Make sure Ollama is running in a separate terminal:
   ```bash
   ollama run llama3
   ```

## 🎬 Usage

Once everything is installed and Ollama is running, execute the pipeline via the terminal:

```bash
python main.py --input "path/to/your/long_video.mp4" --output_dir output_folder
```

Sit back! The terminal will log the pipeline steps (Transcription -> Segmentation -> Reframing -> Subtitling) and your finished vertical Shorts will appear in the output folder.

---
*Note: This tool uses `pyrefly` for type checking. If you'd like to check for strict type adherence, simply run `pyrefly check`.*
