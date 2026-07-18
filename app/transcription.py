import os
import subprocess
import json
from typing import List, Dict, Any

from faster_whisper import WhisperModel


class LocalTranscriber:
    """
    Handles audio extraction and transcription using faster-whisper.
    Target Hardware Constraint: NVIDIA RTX 4050 6GB VRAM (INT8 quantization).
    """

    def __init__(self, model_size: str = "large-v3-turbo", device: str = "cuda", compute_type: str = "int8"):
        """
        Initializes the transcription model.
        """
        print(f"[Transcription] Initializing WhisperModel ({model_size}) on {device} with {compute_type}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def extract_audio(self, video_path: str, output_audio_path: str) -> str:
        """
        Extracts a lightweight WAV file from the source video using FFmpeg.
        """
        print(f"[Transcription] Extracting audio from {video_path} to {output_audio_path}...")
        
        command = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit little-endian
            "-ar", "16000",  # 16kHz sample rate (optimal for whisper)
            "-ac", "1",  # Mono
            output_audio_path
        ]
        
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[Transcription] Audio extraction completed.")
            return output_audio_path
        except subprocess.CalledProcessError as e:
            print(f"[Transcription] FFmpeg error extracting audio: {e}")
            raise RuntimeError("Failed to extract audio using FFmpeg.") from e

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribes the audio file and returns word-level timestamps.
        """
        print(f"[Transcription] Starting transcription for {audio_path}...")
        segments, info = self.model.transcribe(audio_path, word_timestamps=True)
        
        print(f"[Transcription] Detected language '{info.language}' with probability {info.language_probability:.2f}")
        
        result_words = []
        for segment in segments:
            # pyrefly: ignore [not-iterable]
            for word in segment.words:
                result_words.append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end,
                    "probability": word.probability
                })
                
        print(f"[Transcription] Transcription complete. Total words extracted: {len(result_words)}")
        return result_words
