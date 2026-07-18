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
        Processes in 8-minute chunks to avoid overwhelming the LLM's context window.
        """
        all_clips = []
        chunk_duration_limit = 8 * 60.0 # 8 minutes per chunk
        
        # Group transcript into timestamped blocks
        grouped_blocks = []
        current_chunk_start = 0.0
        current_chunk_words = []
        
        for w in transcript_words:
            if not current_chunk_words:
                current_chunk_start = w['start']
            current_chunk_words.append(w['word'])
            
            # Create a block every ~10 seconds or at punctuation
            if w['end'] - current_chunk_start > 10.0 or w['word'].endswith(('.', '?', '!')):
                grouped_blocks.append({
                    "start": current_chunk_start,
                    "end": w['end'],
                    "text": f"[{current_chunk_start:.2f} - {w['end']:.2f}] {' '.join(current_chunk_words)}\n"
                })
                current_chunk_words = []
                
        if current_chunk_words:
             grouped_blocks.append({
                 "start": current_chunk_start,
                 "end": transcript_words[-1]['end'],
                 "text": f"[{current_chunk_start:.2f} - {transcript_words[-1]['end']:.2f}] {' '.join(current_chunk_words)}\n"
             })
             
        # Split blocks into 8-minute chunks
        transcript_chunks = []
        current_text_chunk = ""
        chunk_start_time = 0.0
        
        for block in grouped_blocks:
            if not current_text_chunk:
                chunk_start_time = block['start']
                
            current_text_chunk += block['text']
            
            if block['end'] - chunk_start_time >= chunk_duration_limit:
                transcript_chunks.append(current_text_chunk)
                current_text_chunk = ""
                
        if current_text_chunk:
            transcript_chunks.append(current_text_chunk)
            
        print(f"[Segmentation] Split transcript into {len(transcript_chunks)} manageable chunks for LLM processing.")
        
        for i, chunk_text in enumerate(transcript_chunks):
            print(f"[Segmentation] Processing chunk {i+1}/{len(transcript_chunks)}...")
            prompt = (
                "You are an expert video editor. Analyze the following timestamped transcript and identify the 1 most engaging, cohesive, and standalone story.\n"
                "CRITICAL: The story MUST be between 40 seconds and 90 seconds in total duration. Do NOT select single sentences.\n\n"
                f"Transcript:\n{chunk_text}\n\n"
                "--- END OF TRANSCRIPT ---\n\n"
                "CRITICAL INSTRUCTION:\n"
                "You MUST respond ONLY with a valid JSON array containing exactly 1 object.\n"
                "The object MUST have exactly two keys: 'start_time' and 'end_time' (both must be numbers representing seconds).\n"
                "Example format:\n[{\"start_time\": 10.5, \"end_time\": 55.2}]"
            )
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
            
            try:
                response = requests.post(self.ollama_url, json=payload, timeout=180)
                response.raise_for_status()
                result_json = response.json()
                llm_text = result_json.get("response", "")
                
                clips = self._parse_llm_response(llm_text)
                all_clips.extend(clips)
                
            except requests.exceptions.RequestException as e:
                print(f"[Segmentation] Error communicating with Ollama on chunk {i+1}: {e}")
                
        # Limit to top 3 clips overall to save processing time
        return all_clips[:3]
        
    def _parse_llm_response(self, llm_text: str) -> List[Dict[str, float]]:
        # Parse JSON
        try:
            clips = json.loads(llm_text)
            # Handle cases where LLM returns a single dict instead of a list
            if isinstance(clips, dict):
                if "clips" in clips and isinstance(clips["clips"], list):
                    clips = clips["clips"]
                elif "start_time" in clips or "start" in clips:
                    clips = [clips]
                else:
                    found_list = False
                    for val in clips.values():
                        if isinstance(val, list):
                            clips = val
                            found_list = True
                            break
                    if not found_list:
                        clips = [clips]
            
            if not isinstance(clips, list):
                return []
                
            valid_clips = []
            for c in clips:
                if isinstance(c, dict):
                    start = c.get('start_time', c.get('start'))
                    end = c.get('end_time', c.get('end'))
                    if start is not None and end is not None:
                        # AUTO-PADDING: If LLM returns a tiny clip (e.g. 1 sec), pad it to 45 seconds automatically
                        duration = float(end) - float(start)
                        if duration < 30.0:
                            pad_amount = (45.0 - duration) / 2.0
                            start = max(0.0, float(start) - pad_amount)
                            end = float(end) + pad_amount
                            print(f"[Segmentation] Auto-expanded short clip from {duration:.1f}s to 45s to capture context.")
                            
                        valid_clips.append({'start_time': float(start), 'end_time': float(end)})
            return valid_clips
        except json.JSONDecodeError:
            return []

    def _fallback_parse(self, text: str) -> List[Dict[str, float]]:
        """Fallback method to extract JSON array if Ollama response contains markdown."""
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                valid_clips = []
                if isinstance(parsed, list):
                    for c in parsed:
                        if isinstance(c, dict):
                            start = c.get('start_time', c.get('start'))
                            end = c.get('end_time', c.get('end'))
                            if start is not None and end is not None:
                                valid_clips.append({'start_time': float(start), 'end_time': float(end)})
                        elif isinstance(c, (list, tuple)) and len(c) >= 2:
                            try:
                                valid_clips.append({'start_time': float(c[0]), 'end_time': float(c[1])})
                            except (ValueError, TypeError):
                                pass
                return valid_clips
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
