import cv2
import mediapipe as mp
import subprocess
import os

class VideoReframer:
    """
    Handles dynamic region of interest tracking using MediaPipe Face Detection.
    Converts 16:9 widescreen to 9:16 vertical via a smoothed moving-average crop.
    """
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        # model_selection=1 is for full-range (further faces), 0 is for short-range
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        
    def reframe_video(self, input_video: str, output_video: str):
        """
        Reads frame by frame, tracks faces, smooths coordinates, and crops to 9:16.
        """
        print(f"[Reframing] Starting dynamic reframe for {input_video}...")
        
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {input_video}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Target vertical aspect ratio: 9:16
        target_w = int(orig_h * 9 / 16)
        target_h = orig_h
        
        # 12-frame moving average smoothing as per specs
        window_size = 12
        center_x_history = []
        
        temp_audio = input_video + ".temp_audio.aac"
        temp_video_no_audio = input_video + ".temp_video.mp4"
        
        # Extract audio first
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_video, "-vn", "-acodec", "copy", temp_audio], 
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        # Setup OpenCV VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_no_audio, fourcc, fps, (target_w, target_h))
        
        default_center_x = orig_w / 2
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"[Reframing] Processed {frame_count}/{total_frames} frames...")
                
            # MediaPipe requires RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            center_x = default_center_x
            if results.detections:
                # Take the first prominent face
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                # bbox is normalized relative to frame dimensions
                center_x = (bboxC.xmin + bboxC.width / 2) * orig_w
            else:
                # Fallback to previous center if face lost
                if center_x_history:
                    center_x = center_x_history[-1]
            
            center_x_history.append(center_x)
            if len(center_x_history) > window_size:
                center_x_history.pop(0)
                
            smoothed_center_x = sum(center_x_history) / len(center_x_history)
            
            # Calculate absolute crop boundaries
            x_start = int(smoothed_center_x - target_w / 2)
            
            # Clamp boundaries
            if x_start < 0:
                x_start = 0
            elif x_start + target_w > orig_w:
                x_start = orig_w - target_w
                
            # Crop via standard array index operations
            cropped_frame = frame[:, x_start:x_start + target_w]
            out.write(cropped_frame)
            
        cap.release()
        out.release()
        
        # Multiplex audio back into the reframed video
        print("[Reframing] Multiplexing audio...")
        combine_cmd = [
            "ffmpeg", "-y", 
            "-i", temp_video_no_audio
        ]
        
        # Check if the extracted audio is valid
        has_audio = os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0
        if has_audio:
            combine_cmd.extend(["-i", temp_audio, "-c:v", "copy", "-c:a", "aac"])
        else:
            combine_cmd.extend(["-c:v", "copy"])
            
        combine_cmd.append(output_video)
        
        try:
            subprocess.run(combine_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"[Reframing] Error combining audio and video: {e}")
            raise RuntimeError("FFmpeg multiplexing failed.") from e
        finally:
            # Clean up temp files
            if os.path.exists(temp_video_no_audio):
                os.remove(temp_video_no_audio)
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
                
        print(f"[Reframing] Completed reframing. Output saved to {output_video}")
