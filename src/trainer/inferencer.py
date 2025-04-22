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
            degrader=None,
            batch_transforms=None,
            skip_model_load=False,
    ):
        """
        Initialize the GAN Inferencer.

        Args:
            model_gen (nn.Module): Generator model for super-resolution.
            model_diff (nn.Module): Regular model for super-resolution.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different sets of data.
            save_path (str): path to save generated images and metrics.
            metrics (dict): dict with the definition of metrics for inference.
            batch_transforms (dict[nn.Module] | None): transforms for the batch.
            skip_model_load (bool): if False, load pre-trained checkpoint.
        """
        assert (
                skip_model_load or
                (config.inferencer.model_type == "GAN" and config.inferencer.get("from_pretrained_gen") is not None) or
                (config.inferencer.model_type == "regular" and config.inferencer.get(
                    "from_pretrained_diff") is not None)
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
        self.custom_type = config.inferencer.get("custom_type", None)

        if self.custom_type:
            if hasattr(self, "logger"):
                self.logger.info(f"Using custom model type: {self.custom_type}")
            else:
                print(f"Using custom model type: {self.custom_type}")

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
                self._from_pretrained(pretrained_path_gen=config.inferencer.get("from_pretrained_gen"),
                                      mode="Inference")
            else:
                self._from_pretrained(pretrained_path_diff=config.inferencer.get("from_pretrained_diff"),
                                      mode="Inference")

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

            # Log average PSNR from the logs
            if "PSNR" in logs:
                avg_psnr = logs["PSNR"]
                if hasattr(self, "logger") and self.logger is not None:
                    self.logger.info(f"Average PSNR for {part}: {avg_psnr:.4f}")
                else:
                    print(f"Average PSNR for {part}: {avg_psnr:.4f}")

        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        # Generate low-resolution images
        batch["lr_image"] = self.degrader.process_batch(batch["data_object"])

        # Generate high-resolution images
        if self.config.inferencer.model_type == "GAN":
            batch["gen_output"] = self.model_gen(batch["lr_image"])
        else:  # regular
            custom_type = self.config.inferencer.get("custom_type", None)

            # Handle different return types for different models
            if custom_type == "Swin2SR":
                # The Swin2SR model might return multiple outputs or a tuple
                output = self.model_diff(batch["lr_image"])

                # If it's a tuple, take the first element
                if isinstance(output, tuple):
                    batch["gen_output"] = output[0]
                else:
                    batch["gen_output"] = output
            else:
                # For other models, use standard behavior
                batch["gen_output"] = self.model_diff(batch["lr_image"])

        # Update metrics
        if metrics is not None:
            for met in self.metrics["inference"]:
                metric_value = met(**batch)
                metrics.update(met.name, metric_value)

                # For debugging, print PSNR value every 10 batches
                if met.name == "PSNR" and batch_idx % 10 == 0:
                    if hasattr(self, "logger") and self.logger is not None:
                        self.logger.debug(f"Batch {batch_idx}: PSNR = {metric_value.item():.4f}")

        # Log images at specified intervals
        if self.writer is not None and batch_idx % self.log_step == 0:
            self.writer.set_step(batch_idx, part)
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
        if self.config.inferencer.model_type == "GAN":
            self.model_gen.eval()
        else:  # regular
            self.model_diff.eval()

        self.evaluation_metrics.reset()

        # Create a list to store all PSNR values
        all_psnr_scores = []
        print(self.save_path)
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

                # Get the PSNR value for this batch and add it to our list
                for met in self.metrics["inference"]:
                    if met.name == "PSNR":
                        psnr_value = met(**batch).item()
                        all_psnr_scores.append(psnr_value)

            # Log final metrics
            if self.writer is not None:
                self._log_scalars(self.evaluation_metrics)

            # Save all PSNR scores to a file
            if self.save_path is not None and len(all_psnr_scores) > 0:
                import numpy as np
                scores_path = self.save_path / f"{part}_psnr_scores.txt"
                np.savetxt(str(scores_path), all_psnr_scores)
                if hasattr(self, "logger") and self.logger is not None:
                    self.logger.info(f"Saved {len(all_psnr_scores)} PSNR scores to {scores_path}")
                else:
                    print(f"Saved {len(all_psnr_scores)} PSNR scores to {scores_path}")

        return self.evaluation_metrics.result()