# inferencer.py
import torch
from tqdm.auto import tqdm
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer class for GAN-based super-resolution.
    Used to generate high-resolution images from low-resolution inputs
    and evaluate the model's performance.
    """

    def __init__(
            self,
            model_gen,
            model_diff,
            config,
            device,
            dataloaders,
            save_path,
            metrics=None,
            writer=None,
            logger=None,
            log_step=None,
            degrader = None,
            batch_transforms=None,
            skip_model_load=False,
    ):
        """
        Initialize the GAN Inferencer.

        Args:
            model_gen (nn.Module): Generator model for super-resolution.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different sets of data.
            save_path (str): path to save generated images and metrics.
            metrics (dict): dict with the definition of metrics for inference.
            batch_transforms (dict[nn.Module] | None): transforms for the batch.
            skip_model_load (bool): if False, load pre-trained checkpoint.
        """
        assert (
                skip_model_load or config.inferencer.get("from_pretrained_gen") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer
        self.device = device
        self.model_gen = model_gen
        self.model_diff = model_diff
        self.batch_transforms = batch_transforms
        self.writer = writer
        self.logger = logger
        self.log_step = log_step
        self.degrader = degrader

        # Setup dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # Setup save path
        self.save_path = save_path

        # Setup metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=self.writer,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            if self.config.inferencer.model_type == "GAN":
                self._from_pretrained(pretrained_path_gen = config.inferencer.get("from_pretrained_gen"), mode="Inference")
            else:
                self._from_pretrained(pretrained_path_diff=config.inferencer.get("from_pretrained_diff"), mode="Inference")


    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs


    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Generate high-resolution images from low-resolution inputs
        and compute metrics.

        Args:
            batch_idx (int): current batch index.
            batch (dict): batch data from dataloader.
            metrics (MetricTracker): metrics tracker.
            part (str): partition name for saving.
        Returns:
            batch (dict): processed batch with generated images.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        # Generate low-resolution images
        batch["lr_image"] = self.degrader.process_batch(batch["data_object"])

        # Generate high-resolution images
        batch["gen_output"] = self.model_gen(batch["lr_image"])

        # Update metrics
        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        # Log images at specified intervals
        if self.writer is not None and batch_idx % self.log_step == 0:
            self.writer.set_step(batch_idx, part)  # устанавливаем шаг
            # Берем только первое изображение из батча [0:1]
            self.writer.add_images(f"{part}/original", torch.clamp(batch["data_object"][0:1], 0, 1))
            self.writer.add_images(f"{part}/generated", torch.clamp(batch["gen_output"][0:1], 0, 1))
            self.writer.add_images(f"{part}/low_res", torch.clamp(batch["lr_image"][0:1], 0, 1))

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a partition and save results.

        Args:
            part (str): partition name.
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): computed metrics for the partition.
        """
        self.is_train = False
        self.model_gen.eval()
        self.evaluation_metrics.reset()

        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )
            # Log final metrics
            if self.writer is not None:
                self._log_scalars(self.evaluation_metrics)

        return self.evaluation_metrics.result()