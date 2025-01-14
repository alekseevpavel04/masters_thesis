import json
from datetime import datetime

class ProgressManager:
    @staticmethod
    def save_progress(output_path, processed_frames, total_frames):
        """Save processing progress"""
        progress = {
            'processed_frames': processed_frames,
            'total_frames': total_frames,
            'timestamp': datetime.now().isoformat()
        }
        with open(f"{output_path}.progress", 'w') as f:
            json.dump(progress, f)

    @staticmethod
    def load_progress(output_path):
        """Load processing progress"""
        try:
            with open(f"{output_path}.progress", 'r') as f:
                progress = json.load(f)
            return progress['processed_frames']
        except:
            return 0