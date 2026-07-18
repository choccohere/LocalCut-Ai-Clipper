import os
import json
import re
import subprocess
import requests
from typing import List, Dict, Any

class VideoSegmenter:
    """
    Handles LLM-based narrative extraction and video cutting.
    Connects to local Ollama.
    """
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", model: str = "llama3"):
        self.ollama_url = ollama_url
        self.model = model
        
    def analyze_transcript(self, transcript_words: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Feeds transcript to Ollama and returns a list of target timestamps for clips.
        """
        # Group transcript into timestamped blocks for the LLM
        grouped_transcript = ""
        current_chunk_start = 0.0
        current_chunk_words = []
        for w in transcript_words:
            if not current_chunk_words:
                current_chunk_start = w['start']
            current_chunk_words.append(w['word'])
            
            # Create a block every ~10 seconds or at punctuation
            if w['end'] - current_chunk_start > 10.0 or w['word'].endswith(('.', '?', '!')):
                grouped_transcript += f"[{current_chunk_start:.2f} - {w['end']:.2f}] {' '.join(current_chunk_words)}\n"
                current_chunk_words = []
                
        if current_chunk_words:
             grouped_transcript += f"[{current_chunk_start:.2f} - {transcript_words[-1]['end']:.2f}] {' '.join(current_chunk_words)}\n"
        
        prompt = (
            "You are an expert video editor. Analyze this timestamped transcript and return a JSON array containing the start and end timestamps of the 3 most engaging, cohesive, and standalone stories.\n\n"
            "Respond ONLY with a valid JSON array of objects, with keys 'start_time' and 'end_time' (both numbers representing seconds).\n"
            "Example:\n[{\"start_time\": 10.5, \"end_time\": 55.2}]\n\n"
            f"Transcript:\n{grouped_transcript}"
        )
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json" # Ollama built-in JSON enforcer
        }
        
        print(f"[Segmentation] Sending request to Ollama ({self.model})...")
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=180)
            response.raise_for_status()
            result_json = response.json()
            llm_text = result_json.get("response", "")
            
            # Parse JSON
            try:
                clips = json.loads(llm_text)
                return clips
            except json.JSONDecodeError:
                print("[Segmentation] Warning: Direct JSON parse failed, attempting regex fallback...")
                return self._fallback_parse(llm_text)
                
        except requests.exceptions.RequestException as e:
            print(f"[Segmentation] Error communicating with Ollama: {e}")
            raise RuntimeError("Failed to connect to local Ollama.") from e

    def _fallback_parse(self, text: str) -> List[Dict[str, float]]:
        """Fallback method to extract JSON array if Ollama response contains markdown."""
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        print("[Segmentation] Error: Could not parse LLM output into JSON.")
        return []

    def cut_video(self, video_path: str, output_dir: str, clips: List[Dict[str, float]]) -> List[str]:
        """
        Uses FFmpeg to instantaneously cut video based on timestamps.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_files = []
        for i, clip in enumerate(clips):
            start = clip.get('start_time')
            end = clip.get('end_time')
            
            if start is None or end is None:
                print(f"[Segmentation] Skipping invalid clip format: {clip}")
                continue
                
            out_file = os.path.join(output_dir, f"segment_{i+1}.mp4")
            
            print(f"[Segmentation] Cutting clip {i+1} from {start} to {end}...")
            command = [
                "ffmpeg",
                "-y",
                "-ss", str(start),
                "-i", video_path,
                "-to", str(end - start),
                "-c", "copy",
                out_file
            ]
            
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                output_files.append(out_file)
            except subprocess.CalledProcessError as e:
                print(f"[Segmentation] Error cutting clip {i+1}: {e}")
                
        return output_files
