import os
import sys
import argparse
import time

from app.transcription import LocalTranscriber
from app.segmentation import VideoSegmenter
from app.reframing import VideoReframer
from app.subtitles import SubtitleGenerator

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Local AI Video Clipper")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Keep intermediate files for debugging")
    args = parser.parse_args()
    
    input_video = args.input
    output_dir = args.output_dir
    keep_temp = args.debug
    
    if not os.path.exists(input_video):
        print(f"Error: Input video '{input_video}' not found.")
        sys.exit(1)
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"\n--- Starting Local AI Video Clipper Pipeline ---")
    print(f"Input: {input_video}")
    print(f"Output Directory: {output_dir}\n")
    
    total_start_time = time.time()
    temp_files = []
    
    try:
        # Step 1: Transcription
        print("=== STEP 1: Transcription ===")
        t_start = time.time()
        transcriber = LocalTranscriber()
        audio_path = os.path.join(output_dir, "temp_audio.wav")
        temp_files.append(audio_path)
        
        transcriber.extract_audio(input_video, audio_path)
        transcript_words = transcriber.transcribe(audio_path)
        print(f"Time taken: {time.time() - t_start:.2f} seconds\n")
        
        # Step 2: Segmentation
        print("=== STEP 2: Segmentation ===")
        t_start = time.time()
        segmenter = VideoSegmenter()
        clips = segmenter.analyze_transcript(transcript_words)
        
        if not clips:
            print("Error: No clips identified. Aborting pipeline.")
            sys.exit(1)
            
        print(f"Identified {len(clips)} clips.")
        clip_paths = segmenter.cut_video(input_video, output_dir, clips)
        temp_files.extend(clip_paths) # Original horizontal clips
        print(f"Time taken: {time.time() - t_start:.2f} seconds\n")
        
        # Process each clip
        for i, clip_path in enumerate(clip_paths):
            print(f"=== Processing Clip {i+1} ===")
            clip_start = clips[i].get('start_time', 0.0)
            clip_end = clips[i].get('end_time', 0.0)
            
            # Step 3: Reframing
            print(f"--- Reframing Clip {i+1} ---")
            t_start = time.time()
            reframer = VideoReframer()
            reframed_path = os.path.join(output_dir, f"segment_{i+1}_reframed.mp4")
            temp_files.append(reframed_path)
            
            reframer.reframe_video(clip_path, reframed_path)
            print(f"Reframing time: {time.time() - t_start:.2f} seconds")
            
            # Step 4: Subtitles
            print(f"--- Subtitling Clip {i+1} ---")
            t_start = time.time()
            subtitler = SubtitleGenerator()
            ass_path = os.path.join(output_dir, f"segment_{i+1}.ass")
            temp_files.append(ass_path)
            
            grouped_words = subtitler.group_words(transcript_words, clip_start, clip_end)
            subtitler.generate_ass_file(grouped_words, ass_path)
            
            final_output = os.path.join(output_dir, f"final_clip_{i+1}.mp4")
            subtitler.burn_subtitles(reframed_path, ass_path, final_output)
            print(f"Subtitling time: {time.time() - t_start:.2f} seconds\n")
            print(f"Success: Clip {i+1} saved to {final_output}\n")
            
        print(f"=== Pipeline Completed Successfully in {time.time() - total_start_time:.2f} seconds ===")
        
    except Exception as e:
        print(f"\n!!! Pipeline Failed !!!\nError: {e}")
    finally:
        if not keep_temp:
            print("\nCleaning up temporary files...")
            for f in temp_files:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
            print("Cleanup complete.")
            
if __name__ == "__main__":
    main()
