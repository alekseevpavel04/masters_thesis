import os
import torch
from time import time
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from tqdm import tqdm
from utils import (
    FrameCache,
    setup_logger,
    video_context,
    extract_audio,
    merge_audio,
    ProgressManager,
    FrameProcessor
)
import sys
sys.path.append('../..')
from src.model import RRDBNet


class VideoUpscaler:
    def __init__(self, model_path='model/RealESRGAN_final.pth', batch_size=1, num_workers=4, cache_size=100):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frame_cache = FrameCache(max_size=cache_size)
        self.cache_hits = 0
        self.total_frames = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger()
        self.logger.info(f"Using device: {self.device}")

        self.model = self._load_model(model_path)
        self.frame_processor = FrameProcessor(self.model, self.device)
        self.progress_manager = ProgressManager()

    def _load_model(self, model_path):
        try:
            model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=2)
            loadnet = torch.load(model_path, map_location=self.device, weights_only=True)

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

    def enhance_batch(self, frames):
        enhanced_frames = []
        uncached_frames = []
        uncached_indices = []

        for i, frame in enumerate(frames):
            self.total_frames += 1
            cached_frame = self.frame_cache.get(frame)
            if cached_frame is not None:
                self.cache_hits += 1
                enhanced_frames.append(cached_frame)
            else:
                uncached_frames.append(frame)
                uncached_indices.append(i)

        if uncached_frames:
            output = self.frame_processor.process_batch(uncached_frames, self.frame_cache)
            for i, frame in enumerate(uncached_frames):
                self.frame_cache.put(frame, output[i])
                enhanced_frames.insert(uncached_indices[i], output[i])

        return enhanced_frames

    def process_video(self, input_path, output_path, resume=False):
        # Create a backup of input file and temporary files
        input_base = os.path.splitext(input_path)[0]
        input_ext = os.path.splitext(input_path)[1]
        input_backup = f"{input_base}_original{input_ext}"
        temp_video_path = f"{input_base}_temp{input_ext}"
        temp_audio_path = f"{input_base}_temp.aac"

        try:
            # Create backup of original file if it doesn't exist
            if not os.path.exists(input_backup):
                os.rename(input_path, input_backup)
                os.symlink(input_backup, input_path)  # Create symlink to original

            with video_context(input_backup) as video:  # Use backup for reading
                has_audio = extract_audio(video, temp_audio_path, self.logger)

                fps = video.fps
                total_frames = int(video.duration * fps)
                width, height = video.size

                self._log_video_info(width, height, fps, video.duration, total_frames)

                start_frame = self.progress_manager.load_progress(output_path) if resume else 0
                if start_frame > 0:
                    self.logger.info(f"Resuming from frame {start_frame}")

                # Write to temporary video file
                self._process_frames(video, temp_video_path, width, height, fps,
                                     total_frames, start_frame, output_path)

                # Handle audio merging
                if has_audio:
                    merge_audio(temp_video_path, temp_audio_path, output_path, self.logger)
                else:
                    os.rename(temp_video_path, output_path)

                if os.path.exists(f"{output_path}.progress"):
                    os.remove(f"{output_path}.progress")

        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            # Restore original file if something went wrong
            if os.path.exists(input_backup):
                if os.path.islink(input_path):
                    os.unlink(input_path)
                os.rename(input_backup, input_path)
            raise

        finally:
            # Cleanup temporary files
            self.cache_hits = 0
            self.total_frames = 0
            for temp_file in [temp_video_path, temp_audio_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            # Remove backup if everything went well
            if os.path.exists(input_backup):
                if os.path.islink(input_path):
                    os.unlink(input_path)
                os.rename(input_backup, input_path)

    def _log_video_info(self, width, height, fps, duration, total_frames):
        self.logger.info(f"\nVideo Info:")
        self.logger.info(f"Resolution: {width}x{height}")
        self.logger.info(f"FPS: {fps}")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        self.logger.info(f"Total frames: {total_frames}")
        self.logger.info(f"Output resolution: {width * 2}x{height * 2}\n")

    def _process_frames(self, video, temp_output_path, width, height, fps,
                        total_frames, start_frame, output_path):
        with FFMPEG_VideoWriter(
                temp_output_path,
                (width * 2, height * 2),
                fps,
                ffmpeg_params=['-vcodec', 'libx264', '-crf', '17']
        ) as writer:
            with tqdm(total=total_frames, initial=start_frame,
                      desc="Processing frames", unit="frames") as pbar:
                self._process_frame_batches(video, writer, start_frame, pbar, output_path)

    def _process_frame_batches(self, video, writer, start_frame, pbar, output_path):
        processed_frames = start_frame
        start_time = time()
        fps_buffer = []
        current_batch = []

        for i, frame in enumerate(video.iter_frames()):
            if i < start_frame:
                continue

            frame_start_time = time()
            current_batch.append(frame)

            if len(current_batch) == self.batch_size:
                processed_frames = self._process_and_write_batch(
                    current_batch, writer, processed_frames,
                    frame_start_time, fps_buffer, pbar, start_time, output_path
                )
                current_batch = []

        if current_batch:
            self._process_and_write_batch(
                current_batch, writer, processed_frames,
                frame_start_time, fps_buffer, pbar, start_time, output_path
            )

    def _process_and_write_batch(self, batch, writer, processed_frames,
                                 frame_start_time, fps_buffer, pbar, start_time, output_path):
        enhanced_frames = self.enhance_batch(batch)
        for enhanced_frame in enhanced_frames:
            writer.write_frame(enhanced_frame)
            processed_frames += 1

        frame_time = (time() - frame_start_time) / len(batch)
        fps_buffer.append(1 / frame_time if frame_time > 0 else 0)
        if len(fps_buffer) > 30:
            fps_buffer.pop(0)

        current_fps = sum(fps_buffer) / len(fps_buffer)
        pbar.set_postfix({
            'Elapsed': f"{(time() - start_time):.1f}s",
            'Cache hits': f"{(self.cache_hits / self.total_frames * 100):.1f}%"
        })
        pbar.update(len(batch))

        if processed_frames % 100 == 0:
            self.progress_manager.save_progress(output_path, processed_frames, self.total_frames)

        return processed_frames


def main():
    try:
        model_path = 'model/RealESRGAN_final.pth'
        logger = setup_logger()
        logger.info(f"Model path: {os.path.abspath(model_path)}")

        for path in [model_path, 'input']:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path not found: {path}")

        if not os.path.exists('output'):
            os.makedirs('output')

        input_files = [f for f in os.listdir('input') if f.endswith(('.mkv', '.mp4', '.avi'))]
        if not input_files:
            raise FileNotFoundError("No video files found in /input directory")

        upscaler = VideoUpscaler(
            model_path=model_path,
            batch_size=1,
            num_workers=12,
            cache_size=100
        )

        for i, input_file in enumerate(input_files, 1):
            input_path = os.path.join('input', input_file)
            output_path = os.path.join('output', f'upscaled_{input_file}')
            logger.info(f"\nProcessing video {i}/{len(input_files)}: {input_file}")
            upscaler.process_video(input_path, output_path)

            # Log final statistics after each video
            logger.info("\nVideo processing completed")
            if upscaler.total_frames > 0:
                cache_hit_rate = (upscaler.cache_hits / upscaler.total_frames) * 100
                logger.info(f"Cache statistics:")
                logger.info(f"Total frames processed: {upscaler.total_frames}")
                logger.info(f"Cache hits: {upscaler.cache_hits}")
                logger.info(f"Cache hit rate: {cache_hit_rate:.2f}%")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()