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
sys.path.append('../..')
from src.model import RRDBNet


class VideoUpscaler:
    def __init__(self, model_path='model/RealESRGAN_final.pth', batch_size=1, use_trt = True):
        self.batch_size = batch_size
        self.total_frames = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger()
        self.logger.info(f"Using device: {self.device}")
        if use_trt:
            self.model = self._load_model_trt(model_path)
        else:
            self.model = self._load_model(model_path)
        self.frame_processor = FrameProcessor(
            model=self.model,
            device=self.device,
            tile_size=64,
            overlap=8
        )
        self.progress_manager = ProgressManager()

    def _load_model_trt(self, model_path):
        try:
            # Загрузка базовой модели
            model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=2)
            loadnet = torch.load(model_path, map_location=self.device, weights_only=True)

            if 'params_ema' in loadnet:
                model.load_state_dict(loadnet['params_ema'])
            else:
                model.load_state_dict(loadnet)

            model = model.to(self.device)
            model.eval()

            # Путь к сохраненной TRT модели
            trt_path = model_path.replace('.pth', '_trt.pth')

            # Проверяем существование сохраненной TRT модели
            if os.path.exists(trt_path):
                self.logger.info("Loading cached TRT model...")
                try:
                    from torch2trt import TRTModule
                    model_trt = TRTModule()
                    model_trt.load_state_dict(torch.load(trt_path,weights_only=True))

                    self.logger.info("Cached TRT model loaded successfully")
                    return model_trt

                except Exception as e:
                    self.logger.warning(f"Failed to load cached TRT model: {e}")
                    if os.path.exists(trt_path):
                        os.remove(trt_path)

            # Конвертация в TRT
            self.logger.info("Converting model to TRT...")
            x = torch.randn(1, 3, 64, 64).to(self.device)
            model_trt = torch2trt(
                model,
                [x],
                max_batch_size=442,
                fp16_mode=True,
                max_workspace_size=1 << 25
            )

            # Сохраняем сконвертированную модель
            torch.save(model_trt.state_dict(), trt_path)
            self.logger.info("Model successfully converted to TRT and cached")

            return model_trt

        except Exception as e:
            self.logger.error(f"TRT conversion failed: {e}")
            self.logger.info("Falling back to regular model...")
            return self._load_model(model_path)

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
        self.total_frames += len(frames)
        return self.frame_processor.process_batch(frames)

    def process_video(self, input_path, output_path, resume=False):
        output_dir = os.path.dirname(output_path)
        temp_basename = os.path.splitext(os.path.basename(output_path))[0]
        temp_audio_path = os.path.join(output_dir, f"{temp_basename}_temp.aac")
        temp_video_path = os.path.join(output_dir, f"{temp_basename}_temp.mp4")

        try:
            with video_context(input_path) as video:
                has_audio = extract_audio(video, temp_audio_path, self.logger)

                fps = video.fps
                total_frames = int(video.duration * fps)
                width, height = video.size

                self._log_video_info(width, height, fps, video.duration, total_frames)

                start_frame = self.progress_manager.load_progress(output_path) if resume else 0
                if start_frame > 0:
                    self.logger.info(f"Resuming from frame {start_frame}")

                # Всегда пишем во временный видеофайл с расширением .mp4
                self._process_frames(video, temp_video_path, width, height, fps,
                                  total_frames, start_frame, output_path)

                if has_audio:
                    merge_audio(temp_video_path, temp_audio_path, output_path, self.logger)
                else:
                    os.rename(temp_video_path, output_path)

                # Очистка временных файлов
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                if os.path.exists(temp_video_path) and os.path.exists(output_path):
                    os.remove(temp_video_path)
                if os.path.exists(f"{output_path}.progress"):
                    os.remove(f"{output_path}.progress")

        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            # Очищаем временные файлы в случае ошибки
            for temp_file in [temp_audio_path, temp_video_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            raise

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
        current_batch = []

        for i, frame in enumerate(video.iter_frames()):
            if i < start_frame:
                continue

            current_batch.append(frame)

            if len(current_batch) == self.batch_size:
                processed_frames = self._process_and_write_batch(
                    current_batch, writer, processed_frames, pbar, output_path
                )
                current_batch = []

        if current_batch:
            self._process_and_write_batch(
                current_batch, writer, processed_frames, pbar, output_path
            )

    def _process_and_write_batch(self, batch, writer, processed_frames, pbar, output_path):
        enhanced_frames = self.enhance_batch(batch)
        for enhanced_frame in enhanced_frames:
            writer.write_frame(enhanced_frame)
            processed_frames += 1

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
            use_trt = True
        )

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