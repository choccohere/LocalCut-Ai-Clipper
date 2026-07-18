import os
import subprocess
from typing import List, Dict, Any

class SubtitleGenerator:
    """
    Handles dynamic kinetic typography generation for 9:16 vertical videos.
    Generates Advanced SubStation Alpha (.ass) files and burns them into video.
    """
    
    def __init__(self):
        # Base template for .ass files
        self.ass_template = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,90,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,6,3,5,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def _format_time(self, seconds: float) -> str:
        """Formats seconds to ASS time format H:MM:SS.cs"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds - int(seconds)) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def group_words(self, words: List[Dict[str, Any]], clip_start: float, clip_end: float) -> List[Dict[str, Any]]:
        """
        Filters words to fit within the clip and groups them into 2-3 word segments.
        Adjusts timestamps to be relative to the clip start.
        """
        filtered_words = []
        for w in words:
            if w['end'] >= clip_start and w['start'] <= clip_end:
                # Adjust time to be relative to the clip
                adj_start = max(0.0, w['start'] - clip_start)
                adj_end = min(clip_end - clip_start, w['end'] - clip_start)
                
                filtered_words.append({
                    "word": w['word'],
                    "start": adj_start,
                    "end": adj_end
                })

        grouped = []
        current_group = []
        current_start = 0.0
        
        for w in filtered_words:
            if not current_group:
                current_start = w['start']
                
            current_group.append(w['word'])
            
            # Group by 2-3 words, or if there's a long pause, or punctuation
            if len(current_group) >= 3 or w['word'].endswith(('.', '?', '!', ',')):
                grouped.append({
                    "text": " ".join(current_group),
                    "start": current_start,
                    "end": w['end']
                })
                current_group = []
                
        if current_group:
             grouped.append({
                 "text": " ".join(current_group),
                 "start": current_start,
                 "end": filtered_words[-1]['end']
             })
             
        return grouped

    def generate_ass_file(self, grouped_words: List[Dict[str, Any]], output_path: str):
        """
        Writes the grouped words to an .ass subtitle file.
        """
        print(f"[Subtitles] Generating ASS file at {output_path}...")
        ass_content = self.ass_template
        
        for group in grouped_words:
            start_time = self._format_time(group['start'])
            end_time = self._format_time(group['end'])
            text = group['text']
            
            # Simple kinetic typography: {\an5} centers text, \fscx110\fscy110 adds a slight pop
            styled_text = f"{{\\an5\\t(0,100,\\fscx110\\fscy110)}}{text}"
            
            ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{styled_text}\n"
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ass_content)

    def burn_subtitles(self, video_path: str, ass_path: str, output_path: str):
        """
        Uses FFmpeg to burn the .ass file into the video.
        """
        print(f"[Subtitles] Burning subtitles into {output_path}...")
        
        # FFmpeg ass filter requires escaped paths on Windows
        escaped_ass_path = ass_path.replace("\\", "/").replace(":", "\\:")
        
        command = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vf", f"ass='{escaped_ass_path}'",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[Subtitles] Burn-in complete.")
        except subprocess.CalledProcessError as e:
            print(f"[Subtitles] Error burning subtitles: {e}")
            raise RuntimeError("FFmpeg subtitle burn-in failed.") from e
