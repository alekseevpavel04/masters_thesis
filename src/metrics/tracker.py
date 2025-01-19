from typing import Dict, List, Optional, Union, Any
import pandas as pd
from pandas import Index


class MetricTracker:
    """
    A class for aggregating and tracking metrics across multiple batches.

    This class maintains a DataFrame to store and calculate running statistics
    for various metrics during training or evaluation. It can track total values,
    counts, and running averages for each metric.

    Attributes:
        writer (Optional[Union[WandBWriter, CometMLWriter]]): An experiment tracker
            instance for logging metrics. Currently not used but available for
            future implementations.
        _data (pd.DataFrame): DataFrame storing metric statistics with columns:
            - total: Sum of all values for each metric
            - counts: Number of updates for each metric
            - average: Running average (total/counts) for each metric
    """

    def __init__(
            self,
            *keys: str,
            writer: Optional[Union['WandBWriter', 'CometMLWriter']] = None
    ) -> None:
        """
        Initialize MetricTracker with specified metric keys.

        Args:
            *keys: Variable length list of metric names as strings. These can include
                  names of losses and other evaluation metrics.
            writer: Optional experiment tracker for logging metrics. Can be either
                   WandBWriter or CometMLWriter instance. Currently not used but
                   can be implemented for batch-wise logging.

        Example:
            >>> tracker = MetricTracker('loss', 'accuracy', 'psnr')
            >>> tracker.update('loss', 0.5)
            >>> tracker.avg('loss')
            0.5
        """
        self.writer = writer
        self._data = pd.DataFrame(
            index=keys,
            columns=["total", "counts", "average"]
        )
        self.reset()

    def reset(self) -> None:
        """
        Reset all metric statistics to zero.

        This method is typically called at the end of each epoch to clear
        accumulated statistics before starting a new epoch.

        Note:
            This operation modifies the underlying DataFrame in-place.
        """
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key: str, value: float, n: int = 1) -> None:
        """
        Update statistics for a specified metric.

        This method updates the running statistics (total, count, and average)
        for the specified metric. If the metric doesn't exist, it will be
        automatically added to the tracker.

        Args:
            key: Name of the metric to update
            value: New value to incorporate into the statistics
            n: Weight or count for this update (default: 1)
                Useful for batch processing where each value might
                represent multiple samples

        Note:
            If the metric key doesn't exist, it will be initialized with
            zeros before the update.
        """
        # Ensure the metric exists before trying to update it
        if key not in self._data.index:
            # If not, add the new key with initialized values
            self._data.loc[key] = [0.0, 0, 0.0]

        # Update the metric statistics
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key: str) -> float:
        """
        Get the current average value for a specific metric.

        Args:
            key: Name of the metric whose average is requested

        Returns:
            Current running average for the specified metric

        Note:
            The average is calculated as total/count for the metric.
            Make sure the metric exists before calling this method.
        """
        return self._data.average[key]

    def result(self) -> Dict[str, float]:
        """
        Get current average values for all metrics.

        Returns:
            Dictionary mapping metric names to their current average values

        Example:
            >>> tracker = MetricTracker('loss', 'accuracy')
            >>> tracker.update('loss', 0.5)
            >>> tracker.update('accuracy', 0.95)
            >>> tracker.result()
            {'loss': 0.5, 'accuracy': 0.95}
        """
        return dict(self._data.average)

    def keys(self) -> Index:
        """
        Get names of all metrics currently being tracked.

        Returns:
            Pandas Index containing all metric names in the tracker

        Note:
            The returned Index is a view of the metric names and can be
            used for iteration or checking metric existence.
        """
        return self._data.total.keys()