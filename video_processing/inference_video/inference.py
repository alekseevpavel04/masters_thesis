import os
import cv2
import torch
import numpy as np
from time import time
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from tqdm import tqdm
import sys

sys.path.append('../..')
from src.model import RRDBNet


class VideoUpscaler:
    def __init__(self, model_path='model/RealESRGAN_final.pth'):
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=2)

        # Load model weights
        loadnet = torch.load(model_path, map_location=self.device)
        if 'params_ema' in loadnet:
            self.model.load_state_dict(loadnet['params_ema'])
        else:
            self.model.load_state_dict(loadnet)

        self.model.eval()
        self.model = self.model.to(self.device)

    def enhance_frame(self, img):
        # Convert to tensor and preprocess
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img = img.to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(img)

        # Postprocess
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        return output

    def process_video(self, input_path, output_path):
        # Extract audio first
        print("Extracting audio...")
        temp_audio_path = output_path + '.temp.aac'
        has_audio = False

        try:
            video = VideoFileClip(input_path)
            if video.audio is not None:
                video.audio.write_audiofile(
                    temp_audio_path,
                    codec='aac',  # Explicitly specify codec
                    ffmpeg_params=['-strict', '-2'],  # Allow experimental codecs
                    verbose=False,
                    logger=None
                )
                has_audio = True
            video.close()
        except Exception as e:
            print(f"Warning: Could not extract audio: {str(e)}")
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            has_audio = False

        # Process video
        print("Processing video...")
        video = VideoFileClip(input_path)
        fps = video.fps
        total_frames = int(video.duration * fps)

        print(f"\nVideo Info:")
        print(f"Resolution: {video.size[0]}x{video.size[1]}")
        print(f"FPS: {fps}")
        print(f"Duration: {video.duration:.2f} seconds")
        print(f"Total frames: {total_frames}")
        print(f"Output resolution: {video.size[0] * 2}x{video.size[1] * 2}\n")

        # Create temporary path for video without audio
        temp_output_path = output_path + '.temp.mp4'

        # Initialize writer
        width, height = video.size
        writer = FFMPEG_VideoWriter(
            temp_output_path,
            (width * 2, height * 2),  # 2x upscale
            fps,
            ffmpeg_params=['-vcodec', 'libx264', '-crf', '17']
        )

        # Initialize progress bar
        pbar = tqdm(total=total_frames,
                    desc="Processing frames",
                    unit="frames",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

        processed_frames = 0
        start_time = time()
        fps_buffer = []

        try:
            # Process frames
            for frame in video.iter_frames():
                frame_start_time = time()

                # Process frame
                enhanced_frame = self.enhance_frame(frame)
                writer.write_frame(enhanced_frame)

                # Update progress
                processed_frames += 1
                frame_time = time() - frame_start_time
                fps_buffer.append(1 / frame_time if frame_time > 0 else 0)

                # Keep only last 30 frames for FPS calculation
                if len(fps_buffer) > 30:
                    fps_buffer.pop(0)

                current_fps = sum(fps_buffer) / len(fps_buffer)

                # Update progress bar with FPS
                pbar.set_postfix({
                    'FPS': f"{current_fps:.1f}",
                    'Elapsed': f"{(time() - start_time):.1f}s"
                })
                pbar.update(1)

        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            raise e

        finally:
            pbar.close()
            writer.close()
            video.close()

        # Combine video with audio if available
        if has_audio:
            print("\nMerging video with audio...")
            import subprocess

            try:
                cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_output_path,
                    '-i', temp_audio_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-strict', '-2',  # Allow experimental codecs
                    output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                print("Audio merged successfully")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to merge audio: {e.stderr.decode()}")
                # If audio merging fails, just use the video without audio
                os.rename(temp_output_path, output_path)
        else:
            # If no audio in original, just rename the temp file
            os.rename(temp_output_path, output_path)

        # Cleanup temporary files
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        end_time = time()
        print(f"\nVideo processing completed in {end_time - start_time:.2f} seconds")
        print(f"Average FPS: {processed_frames / (end_time - start_time):.2f}")


def main():
    model_path = 'model/RealESRGAN_final.pth'
    print(f"Model path: {os.path.abspath(model_path)}")

    # Check directories exist
    if not os.path.exists('model/RealESRGAN_final.pth'):
        raise FileNotFoundError("Model file not found at /model/RealESRGAN_final.pth")

    if not os.path.exists('input'):
        raise FileNotFoundError("Input directory not found at /input")

    if not os.path.exists('output'):
        os.makedirs('output')

    # Find all mkv files in input directory
    input_files = [f for f in os.listdir('input') if f.endswith('.mkv')]

    if not input_files:
        raise FileNotFoundError("No .mkv files found in /input directory")

    # Initialize upscaler
    upscaler = VideoUpscaler()

    # Process each video
    for i, input_file in enumerate(input_files, 1):
        input_path = os.path.join('input', input_file)
        output_path = os.path.join('output', f'upscaled_{input_file}')

        print(f"\nProcessing video {i}/{len(input_files)}: {input_file}")
        upscaler.process_video(input_path, output_path)


if __name__ == "__main__":
    main()