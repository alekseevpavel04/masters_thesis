import os
import torch
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from tqdm import tqdm
from utils import (
    setup_logger,
    video_context,
    extract_audio,
    merge_audio,
    ProgressManager,
    FrameProcessor
)
from torch2trt import torch2trt
import sys

# Add the parent directory to the system path to import the model
sys.path.append('../..')
from src.model import RRDBNet


class VideoUpscaler:
    """
    A class for upscaling video content using AI-based super-resolution.

    This class implements video upscaling functionality using the RRDBNet model,
    with optional TensorRT optimization for improved performance.

    Attributes:
        batch_size (int): Number of frames to process simultaneously.
        total_frames (int): Counter for total processed frames.
        device (torch.device): Device used for computation (CPU or CUDA).
        logger: Logger instance for tracking operations.
        model: The loaded AI model (either regular PyTorch or TensorRT optimized).
        frame_processor (FrameProcessor): Handler for frame-level processing.
        progress_manager (ProgressManager): Tracks processing progress.
    """

    def __init__(
            self,
            model_path: str = 'model/RealESRGAN_final.pth',
            batch_size: int = 1,
            use_trt: bool = True
    ):
        """
        Initialize the VideoUpscaler with specified parameters.

        Args:
            model_path (str): Path to the pretrained model weights.
            batch_size (int): Number of frames to process in parallel.
            use_trt (bool): Whether to use TensorRT optimization.
        """
        self.batch_size = batch_size
        self.total_frames = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger()
        self.logger.info(f"Using device: {self.device}")

        # Load the model with or without TensorRT optimization
        self.model = self._load_model_trt(model_path) if use_trt else self._load_model(model_path)
        self.frame_processor = FrameProcessor(
            model=self.model,
            device=self.device,
            tile_size=128,
            overlap=8
        )
        self.progress_manager = ProgressManager()

    def _load_model_trt(self, model_path: str) -> torch.nn.Module:
        """
        Load and optimize the model using TensorRT.

        Args:
            model_path (str): Path to the model weights.

        Returns:
            torch.nn.Module: TensorRT optimized model or regular PyTorch model if optimization fails.

        Raises:
            Exception: If model loading fails.
        """
        try:
            # Load base model
            model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=2)
            loadnet = torch.load(model_path, map_location=self.device, weights_only=True)

            # Load model weights
            if 'params_ema' in loadnet:
                model.load_state_dict(loadnet['params_ema'])
            else:
                model.load_state_dict(loadnet)

            model = model.to(self.device)
            model.eval()

            # Path for cached TRT model
            trt_path = model_path.replace('.pth', '_trt.pth')

            # Try loading cached TRT model
            if os.path.exists(trt_path):
                self.logger.info("Loading cached TRT model...")
                try:
                    from torch2trt import TRTModule
                    model_trt = TRTModule()
                    model_trt.load_state_dict(torch.load(trt_path, weights_only=True))
                    self.logger.info("Cached TRT model loaded successfully")
                    return model_trt
                except Exception as e:
                    self.logger.warning(f"Failed to load cached TRT model: {e}")
                    if os.path.exists(trt_path):
                        os.remove(trt_path)

            # Convert to TRT
            self.logger.info("Converting model to TRT...")
            x = torch.randn(1, 3, 128, 128).to(self.device)
            model_trt = torch2trt(
                model,
                [x],
                max_batch_size=96,
                fp16_mode=True,
                max_workspace_size=1 << 30,
                use_onnx=True
            )

            # Save converted model
            torch.save(model_trt.state_dict(), trt_path)
            self.logger.info("Model successfully converted to TRT and cached")

            return model_trt

        except Exception as e:
            self.logger.error(f"TRT conversion failed: {e}")
            self.logger.info("Falling back to regular model...")
            return self._load_model(model_path)

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load the regular PyTorch model.

        Args:
            model_path (str): Path to the model weights.

        Returns:
            torch.nn.Module: Loaded PyTorch model.

        Raises:
            Exception: If model loading fails.
        """
        try:
            model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=2)
            loadnet = torch.load(model_path, map_location=self.device, weights_only=True)

            # Load model weights
            if 'params_ema' in loadnet:
                model.load_state_dict(loadnet['params_ema'])
            else:
                model.load_state_dict(loadnet)

            model.eval()
            model = model.to(self.device)
            self.logger.info("Model loaded successfully")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def enhance_batch(self, frames: list) -> list:
        """
        Process a batch of frames through the upscaling model.

        Args:
            frames (list): List of input frames to process.

        Returns:
            list: List of enhanced frames.
        """
        self.total_frames += len(frames)
        return self.frame_processor.process_batch(frames)

    def process_video(self, input_path: str, output_path: str, resume: bool = False):
        """
        Process an entire video file, upscaling all frames.

        Args:
            input_path (str): Path to input video file.
            output_path (str): Path for output video file.
            resume (bool): Whether to resume from a previous processing attempt.

        Raises:
            Exception: If video processing fails.
        """
        output_dir = os.path.dirname(output_path)
        temp_basename = os.path.splitext(os.path.basename(output_path))[0]
        temp_audio_path = os.path.join(output_dir, f"{temp_basename}_temp.aac")
        temp_video_path = os.path.join(output_dir, f"{temp_basename}_temp.mp4")

        try:
            with video_context(input_path) as video:
                # Extract audio from the video if it exists
                has_audio = extract_audio(video, temp_audio_path, self.logger)

                fps = video.fps
                total_frames = int(video.duration * fps)
                width, height = video.size

                # Log video information
                self._log_video_info(width, height, fps, video.duration, total_frames)

                # Resume from a specific frame if requested
                start_frame = self.progress_manager.load_progress(output_path) if resume else 0
                if start_frame > 0:
                    self.logger.info(f"Resuming from frame {start_frame}")

                # Process all frames in the video
                self._process_frames(
                    video,
                    temp_video_path,
                    width,
                    height,
                    fps,
                    total_frames,
                    start_frame,
                    output_path
                )

                # Merge audio back if it was extracted
                if has_audio:
                    merge_audio(temp_video_path, temp_audio_path, output_path, self.logger)
                else:
                    os.rename(temp_video_path, output_path)

                # Cleanup temporary files
                for temp_file in [temp_audio_path, temp_video_path, f"{output_path}.progress"]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            # Cleanup on error
            for temp_file in [temp_audio_path, temp_video_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            raise

    def _log_video_info(
            self,
            width: int,
            height: int,
            fps: float,
            duration: float,
            total_frames: int
    ):
        """
        Log video information to the logger.

        Args:
            width (int): Video width in pixels.
            height (int): Video height in pixels.
            fps (float): Frames per second.
            duration (float): Video duration in seconds.
            total_frames (int): Total number of frames.
        """
        self.logger.info(f"\nVideo Info:")
        self.logger.info(f"Resolution: {width}x{height}")
        self.logger.info(f"FPS: {fps}")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        self.logger.info(f"Total frames: {total_frames}")
        self.logger.info(f"Output resolution: {width * 2}x{height * 2}\n")

    def _process_frames(
            self,
            video,
            temp_output_path: str,
            width: int,
            height: int,
            fps: float,
            total_frames: int,
            start_frame: int,
            output_path: str
    ):
        """
        Process all frames in the video.

        Args:
            video: Video object to process.
            temp_output_path (str): Path for temporary output file.
            width (int): Video width.
            height (int): Video height.
            fps (float): Frames per second.
            total_frames (int): Total number of frames.
            start_frame (int): Frame to start processing from.
            output_path (str): Final output path.
        """
        with FFMPEG_VideoWriter(
                temp_output_path,
                (width * 2, height * 2),
                fps,
                ffmpeg_params=['-vcodec', 'libx264', '-crf', '17']
        ) as writer:
            with tqdm(
                    total=total_frames,
                    initial=start_frame,
                    desc="Processing frames",
                    unit="frames"
            ) as pbar:
                self._process_frame_batches(video, writer, start_frame, pbar, output_path)

    def _process_frame_batches(
            self,
            video,
            writer: FFMPEG_VideoWriter,
            start_frame: int,
            pbar: tqdm,
            output_path: str
    ):
        """
        Process video frames in batches.

        Args:
            video: Video object to process.
            writer (FFMPEG_VideoWriter): Video writer object.
            start_frame (int): Frame to start from.
            pbar (tqdm): Progress bar object.
            output_path (str): Path for saving progress.
        """
        processed_frames = start_frame
        current_batch = []

        for i, frame in enumerate(video.iter_frames()):
            if i < start_frame:
                continue

            current_batch.append(frame)

            if len(current_batch) == self.batch_size:
                processed_frames = self._process_and_write_batch(
                    current_batch,
                    writer,
                    processed_frames,
                    pbar,
                    output_path
                )
                current_batch = []

        # Process remaining frames
        if current_batch:
            self._process_and_write_batch(
                current_batch,
                writer,
                processed_frames,
                pbar,
                output_path
            )

    def _process_and_write_batch(
            self,
            batch: list,
            writer: FFMPEG_VideoWriter,
            processed_frames: int,
            pbar: tqdm,
            output_path: str
    ) -> int:
        """
        Process and write a batch of frames.

        Args:
            batch (list): List of frames to process.
            writer (FFMPEG_VideoWriter): Video writer object.
            processed_frames (int): Number of frames processed so far.
            pbar (tqdm): Progress bar object.
            output_path (str): Path for saving progress.

        Returns:
            int: Updated count of processed frames.
        """
        enhanced_frames = self.enhance_batch(batch)
        for enhanced_frame in enhanced_frames:
            writer.write_frame(enhanced_frame)
            processed_frames += 1

        pbar.update(len(batch))

        if processed_frames % 100 == 0:
            self.progress_manager.save_progress(output_path, processed_frames, self.total_frames)

        return processed_frames


def main():
    """
    Main function to run the video upscaling process.
    Processes all video files in the input directory and saves results to the output directory.

    Raises:
        FileNotFoundError: If required paths or files are not found.
        Exception: For any other processing errors.
    """
    try:
        model_path = 'model/RealESRGAN_final.pth'
        logger = setup_logger()
        logger.info(f"Model path: {os.path.abspath(model_path)}")

        # Verify required paths exist
        for path in [model_path, 'input']:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path not found: {path}")

        # Create output directory if it doesn't exist
        if not os.path.exists('output'):
            os.makedirs('output')

        # Get list of video files to process
        input_files = [f for f in os.listdir('input')
                       if f.endswith(('.mkv', '.mp4', '.avi'))]
        if not input_files:
            raise FileNotFoundError("No video files found in /input directory")

        # Initialize upscaler
        upscaler = VideoUpscaler(
            model_path=model_path,
            batch_size=2,
            use_trt=True
        )

        # Process each video file
        for i, input_file in enumerate(input_files, 1):
            input_path = os.path.join('input', input_file)
            output_path = os.path.join('output', f'upscaled_{input_file}')
            logger.info(f"\nProcessing video {i}/{len(input_files)}: {input_file}")
            upscaler.process_video(input_path, output_path)
            logger.info("\nVideo processing completed")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()