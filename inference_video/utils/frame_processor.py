import torch
import torch.backends.cudnn as cudnn
import numpy as np


class TiledProcessor:
    """
    A class for processing images in tiles, handling overlapping regions for seamless merging.

    Attributes:
        tile_size (int): Size of each tile.
        overlap (int): Overlap between adjacent tiles.
    """

    def __init__(self, tile_size=64, overlap=8):
        """
        Initialize the TiledProcessor with specified tile size and overlap.

        Args:
            tile_size (int): Size of each tile.
            overlap (int): Overlap between adjacent tiles.
        """
        self.tile_size = tile_size
        self.overlap = overlap

    def split_image(self, image):
        """
        Split an image into overlapping tiles with strict edge alignment.

        Args:
            image (numpy.ndarray): Input image to be split into tiles.

        Returns:
            list: List of tiles.
            list: List of tile positions.
            tuple: Number of tiles in height and width.
        """
        h, w = image.shape[:2]

        # Calculate effective stride (tile_size - overlap)
        stride = self.tile_size - self.overlap

        # Calculate number of tiles needed
        n_h = max(1, (h + stride - 1) // stride)
        n_w = max(1, (w + stride - 1) // stride)

        tiles = []
        positions = []

        for i in range(n_h):
            for j in range(n_w):
                # For middle tiles, use normal stride
                left = j * stride
                top = i * stride

                # For edge tiles, align strictly with image boundary
                if i == n_h - 1:  # Last row
                    top = max(0, h - self.tile_size)
                if j == n_w - 1:  # Last column
                    left = max(0, w - self.tile_size)

                # Special case for small images
                if h < self.tile_size:
                    top = 0
                if w < self.tile_size:
                    left = 0

                # Extract tile
                tile = image[top:min(top + self.tile_size, h),
                       left:min(left + self.tile_size, w)]

                # Pad if necessary (for edge tiles that might be smaller)
                if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                    padded_tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile

                tiles.append(tile)
                positions.append((top, left))

        return tiles, positions, (n_h, n_w)

    def merge_tiles(self, tiles, positions, original_shape, scale_factor=2):
        """
        Merge processed tiles with strict edge handling.

        Args:
            tiles (list): List of processed tiles.
            positions (list): List of tile positions.
            original_shape (tuple): Original shape of the image.
            scale_factor (int): Scaling factor for the output.

        Returns:
            numpy.ndarray: Merged image.
        """
        h, w = original_shape[:2]
        out_h, out_w = h * scale_factor, w * scale_factor
        output = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weights = np.zeros((out_h, out_w, 3), dtype=np.float32)  # Changed to match output dimensions

        # Create weight mask for blending
        weight_mask = np.ones((self.tile_size * scale_factor,
                               self.tile_size * scale_factor, 3), dtype=np.float32)  # Changed to 3 channels

        # Apply smoother blending only for overlapping regions
        overlap_scaled = self.overlap * scale_factor
        if overlap_scaled > 0:
            for i in range(overlap_scaled):
                # Using smooth cubic interpolation for weight transition
                factor = (i / overlap_scaled) * (i / overlap_scaled) * (3 - 2 * i / overlap_scaled)
                weight_mask[i, :, :] *= factor
                weight_mask[-i - 1, :, :] *= factor
                weight_mask[:, i, :] *= factor
                weight_mask[:, -i - 1, :] *= factor

        for tile, (top, left) in zip(tiles, positions):
            # Scale positions for output resolution
            top_scaled = top * scale_factor
            left_scaled = left * scale_factor

            # Calculate output tile size
            out_tile_h = min(self.tile_size * scale_factor, out_h - top_scaled)
            out_tile_w = min(self.tile_size * scale_factor, out_w - left_scaled)

            # Handle edge cases
            if out_tile_h <= 0 or out_tile_w <= 0:
                continue

            # Ensure tile data is in float32 range [0, 1]
            tile = np.clip(tile, 0, 1)

            # Add tile to output with weight mask
            output_slice = output[top_scaled:top_scaled + out_tile_h,
                           left_scaled:left_scaled + out_tile_w]
            weights_slice = weights[top_scaled:top_scaled + out_tile_h,
                            left_scaled:left_scaled + out_tile_w]

            tile_scaled = tile[:out_tile_h, :out_tile_w]
            mask_slice = weight_mask[:out_tile_h, :out_tile_w]

            output_slice += tile_scaled * mask_slice
            weights_slice += mask_slice

        # Avoid division by zero
        mask = (weights > 1e-8)
        np.divide(output, weights, out=output, where=mask)

        # Convert to uint8 safely
        return np.clip(output * 255.0, 0, 255).astype(np.uint8)


class FrameProcessor:
    """
    A class for processing video frames using a deep learning model.

    Attributes:
        model: The deep learning model for frame processing.
        device: The device (CPU or GPU) used for computation.
        tiled_processor: Instance of TiledProcessor for handling tiled processing.
        stream: CUDA stream for asynchronous processing (if using GPU).
        memory_format: Memory format for tensor storage.
    """

    def __init__(self, model, device, tile_size=64, overlap=8):
        """
        Initialize the FrameProcessor with the model, device, and tiling parameters.

        Args:
            model: The deep learning model for frame processing.
            device: The device (CPU or GPU) used for computation.
            tile_size (int): Size of each tile.
            overlap (int): Overlap between adjacent tiles.
        """
        self.model = model
        self.device = device
        self.tiled_processor = TiledProcessor(tile_size=tile_size, overlap=overlap)

        if self.device.type == 'cuda':
            cudnn.benchmark = True
            cudnn.deterministic = False
            cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

        self.memory_format = torch.channels_last if device.type == 'cuda' else torch.contiguous_format
        self.model = self.model.to(memory_format=self.memory_format)

    def process_batch(self, frames):
        """
        Process a batch of frames using the model.

        Args:
            frames (list): List of input frames to process.

        Returns:
            list: List of processed frames.
        """
        # Split all frames into tiles
        all_tiles = []
        all_positions = []
        frame_tile_counts = []

        for frame in frames:
            tiles, positions, _ = self.tiled_processor.split_image(frame)
            all_tiles.extend(tiles)
            all_positions.append(positions)
            frame_tile_counts.append(len(tiles))

        # Convert tiles to tensor and normalize
        tiles_tensor = torch.from_numpy(
            np.array(all_tiles).astype(np.float32) / 255.
        ).permute(0, 3, 1, 2)

        if self.device.type == 'cuda':
            tiles_tensor = tiles_tensor.to(memory_format=self.memory_format)

        tiles_tensor = tiles_tensor.to(self.device)

        # Process tiles using the model
        with torch.cuda.stream(self.stream) if self.stream else nullcontext():
            with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                with torch.no_grad():
                    output = self.model(tiles_tensor)

            processed_tiles = output.permute(0, 2, 3, 1).cpu().numpy()
            processed_tiles = processed_tiles.clip(0, 1)

            if self.device.type == 'cuda':
                self.stream.synchronize()
                torch.cuda.empty_cache()

        # Merge processed tiles back into frames
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

            processed_frames.append(merged_frame)
            start_idx += n_tiles

        return processed_frames