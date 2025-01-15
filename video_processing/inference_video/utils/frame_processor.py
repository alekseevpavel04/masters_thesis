import torch
import torch.backends.cudnn as cudnn
import numpy as np


class TiledProcessor:
    def __init__(self, tile_size=64, overlap=8):
        self.tile_size = tile_size
        self.overlap = overlap

    def split_image(self, image):
        """Split image into overlapping tiles"""
        h, w = image.shape[:2]

        # Calculate effective stride (tile_size - overlap)
        stride = self.tile_size - self.overlap

        # Calculate number of tiles in each dimension
        n_h = (h - self.overlap) // stride
        n_w = (w - self.overlap) // stride

        # Adjust n_h and n_w to ensure coverage of the entire image
        if h > n_h * stride + self.overlap:
            n_h += 1
        if w > n_w * stride + self.overlap:
            n_w += 1

        tiles = []
        positions = []

        for i in range(n_h):
            for j in range(n_w):
                # Calculate tile position
                top = min(i * stride, h - self.tile_size)
                left = min(j * stride, w - self.tile_size)

                # Extract tile
                tile = image[top:top + self.tile_size, left:left + self.tile_size]
                tiles.append(tile)
                positions.append((top, left))

        return tiles, positions, (n_h, n_w)

    def merge_tiles(self, tiles, positions, original_shape, scale_factor=2):
        """Merge processed tiles back into a single image"""
        h, w = original_shape[:2]
        out_h, out_w = h * scale_factor, w * scale_factor
        output = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weights = np.zeros((out_h, out_w, 1), dtype=np.float32)

        # Create weight mask for blending
        weight_mask = np.ones((self.tile_size * scale_factor, self.tile_size * scale_factor, 1))
        for i in range(self.overlap * scale_factor):
            weight_mask[i, :, 0] *= i / (self.overlap * scale_factor)
            weight_mask[-i - 1, :, 0] *= i / (self.overlap * scale_factor)
            weight_mask[:, i, 0] *= i / (self.overlap * scale_factor)
            weight_mask[:, -i - 1, 0] *= i / (self.overlap * scale_factor)

        for tile, (top, left) in zip(tiles, positions):
            top *= scale_factor
            left *= scale_factor

            # Calculate position for upscaled tile
            h_end = min(top + self.tile_size * scale_factor, out_h)
            w_end = min(left + self.tile_size * scale_factor, out_w)
            h_start = top
            w_start = left

            # Get the region of the tile that fits
            tile_h = h_end - h_start
            tile_w = w_end - w_start

            # Add tile to output with weight mask
            output[h_start:h_end, w_start:w_end] += tile[:tile_h, :tile_w] * weight_mask[:tile_h, :tile_w]
            weights[h_start:h_end, w_start:w_end] += weight_mask[:tile_h, :tile_w]

        # Normalize by weights to blend tiles
        output = np.divide(output, weights, where=weights != 0)
        return output


class FrameProcessor:
    def __init__(self, model, device, tile_size=64, overlap=8):
        self.model = model
        self.device = device
        self.tiled_processor = TiledProcessor(tile_size=tile_size, overlap=overlap)

        if self.device.type == 'cuda':
            cudnn.benchmark = True
            cudnn.deterministic = False
            cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

        self.memory_format = torch.channels_last if device.type == 'cuda' else torch.contiguous_format
        self.model = self.model.to(memory_format=self.memory_format)

    def process_batch(self, frames):
        # Разбиваем все кадры на тайлы одновременно
        all_tiles = []
        all_positions = []
        frame_tile_counts = []  # Сохраняем количество тайлов для каждого кадра

        # Собираем все тайлы со всех кадров
        for frame in frames:
            tiles, positions, _ = self.tiled_processor.split_image(frame)
            all_tiles.extend(tiles)
            all_positions.append(positions)
            frame_tile_counts.append(len(tiles))

        # Преобразуем все тайлы в тензор и обрабатываем за один проход
        tiles_tensor = torch.from_numpy(
            np.array(all_tiles).astype(np.float32) / 255.
        ).permute(0, 3, 1, 2)

        if self.device.type == 'cuda':
            tiles_tensor = tiles_tensor.to(memory_format=self.memory_format)

        tiles_tensor = tiles_tensor.to(self.device)

        # Обрабатываем все тайлы одним батчем
        with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
            with torch.no_grad():
                output = self.model(tiles_tensor)

        processed_tiles = output.permute(0, 2, 3, 1).cpu().numpy()
        processed_tiles = processed_tiles.clip(0, 1)

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Собираем кадры обратно
        processed_frames = []
        start_idx = 0

        for frame_idx, frame in enumerate(frames):
            n_tiles = frame_tile_counts[frame_idx]
            frame_tiles = processed_tiles[start_idx:start_idx + n_tiles]

            merged_frame = self.tiled_processor.merge_tiles(
                frame_tiles,
                all_positions[frame_idx],
                frame.shape
            )

            merged_frame = (merged_frame * 255).clip(0, 255).astype(np.uint8)
            processed_frames.append(merged_frame)
            start_idx += n_tiles

        return processed_frames