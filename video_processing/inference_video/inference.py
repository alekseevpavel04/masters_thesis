import os
import cv2
import torch
import numpy as np
from time import time
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import sys
sys.path.append('../..')
from src.model import RRDBNet

class VideoUpscaler:
    def __init__(self, model_path='model/RealESRGAN_final.pth'):
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        # Load video
        video = VideoFileClip(input_path)
        fps = video.fps
        total_frames = int(video.duration * fps)

        # Initialize writer
        width, height = video.size
        writer = FFMPEG_VideoWriter(
            output_path,
            (width * 2, height * 2),  # 2x upscale
            fps,
            ffmpeg_params=['-vcodec', 'libx264', '-crf', '17']
        )

        print(f"Processing video with {total_frames} frames...")
        start_time = time()

        try:
            # Process frames
            for i, frame in enumerate(video.iter_frames()):
                if i % 10 == 0:
                    elapsed_time = time() - start_time
                    frames_per_second = i / elapsed_time if elapsed_time > 0 else 0
                    print(f"Processing frame {i}/{total_frames} ({frames_per_second:.2f} fps)")

                # Process frame
                enhanced_frame = self.enhance_frame(frame)

                # Write frame
                writer.write_frame(enhanced_frame)

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise e

        finally:
            # Cleanup
            writer.close()
            video.close()

        end_time = time()
        print(f"Video processing completed in {end_time - start_time:.2f} seconds")


def main():
    import os
    model_path = 'model/RealESRGAN_final.pth'
    print(os.path.abspath(model_path))
    # Check directories exist
    if not os.path.exists('model/RealESRGAN_final.pth'):
        raise FileNotFoundError("Model file not found at /model/RealESRGAN_final.pth")

    if not os.path.exists('input'):
        raise FileNotFoundError("Input directory not found at /input")

    if not os.path.exists('output'):
        os.makedirs('/output')

    # Find all mkv files in input directory
    input_files = [f for f in os.listdir('input') if f.endswith('.mkv')]

    if not input_files:
        raise FileNotFoundError("No .mkv files found in /input directory")

    # Initialize upscaler
    upscaler = VideoUpscaler()

    # Process each video
    for input_file in input_files:
        input_path = os.path.join('input', input_file)
        output_path = os.path.join('output', f'upscaled_{input_file}')

        print(f"\nProcessing {input_file}...")
        upscaler.process_video(input_path, output_path)


if __name__ == "__main__":
    main()