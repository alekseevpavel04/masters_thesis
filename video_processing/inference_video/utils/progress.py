import json
from datetime import datetime


class ProgressManager:
    """
    A class for managing and saving the progress of video processing.
    """

    @staticmethod
    def save_progress(output_path, processed_frames, total_frames):
        """
        Save the current progress of video processing.

        Args:
            output_path (str): Path to the output video file.
            processed_frames (int): Number of frames processed so far.
            total_frames (int): Total number of frames in the video.
        """
        progress = {
            'processed_frames': processed_frames,
            'total_frames': total_frames,
            'timestamp': datetime.now().isoformat()
        }
        with open(f"{output_path}.progress", 'w') as f:
            json.dump(progress, f)

    @staticmethod
    def load_progress(output_path):
        """
        Load the saved progress of video processing.

        Args:
            output_path (str): Path to the output video file.

        Returns:
            int: Number of frames processed so far, or 0 if no progress is found.
        """
        try:
            with open(f"{output_path}.progress", 'r') as f:
                progress = json.load(f)
            return progress['processed_frames']
        except:
            return 0