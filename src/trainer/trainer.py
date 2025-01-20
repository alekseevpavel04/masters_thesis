from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch


class Trainer(BaseTrainer):
    def __init__(self, *args, gradient_accumulation_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.content_weight = 1.0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.current_accumulation_step = 0
        self.last_metrics = {}

    def _reset_accumulation_step(self):
        """Reset gradient accumulation counter"""
        self.current_accumulation_step = 0

    def _train_epoch(self, epoch):
        self._reset_accumulation_step()
        try:
            result = super()._train_epoch(epoch)
            torch.cuda.empty_cache()
            return result
        except Exception as e:
            torch.cuda.empty_cache()
            raise e

    def process_batch(self, batch, metrics: MetricTracker):
        try:
            batch = self.move_batch_to_device(batch)
            batch = self.transform_batch(batch)

            metric_funcs = self.metrics["inference"]
            if self.is_train:
                metric_funcs = self.metrics["train"]

                # Train Discriminator
                for _ in range(self.disc_steps):
                    if self.current_accumulation_step == 0:
                        self.optimizer_disc.zero_grad()

                    # Generate low resolution images and store in batch
                    batch["lr_image"] = self.degrader.process_batch(batch["data_object"])

                    # Generate fake samples
                    with torch.no_grad():
                        batch["gen_output"] = self.model_gen(batch["lr_image"])

                    # Get discriminator predictions
                    disc_fake = self.model_disc(batch["gen_output"].detach())
                    disc_real = self.model_disc(batch["data_object"])

                    # Calculate discriminator losses
                    batch["disc_loss"] = self.criterion_disc(disc_fake, disc_real, batch)

                    # Explicitly delete intermediate tensors
                    del disc_fake, disc_real
                    torch.cuda.empty_cache()

                    # Scale and backward
                    scaled_disc_loss = batch["disc_loss"] / self.gradient_accumulation_steps
                    scaled_disc_loss.backward()
                    del scaled_disc_loss

                    self._clip_grad_norm(self.model_disc)

                    if (self.current_accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer_disc.step()

                # Train Generator
                if self.current_accumulation_step == 0:
                    self.optimizer_gen.zero_grad()

                # Reuse or update lr_image
                batch["lr_image"] = self.degrader.process_batch(batch["data_object"])

                # Generate fake images
                batch["gen_output"] = self.model_gen(batch["lr_image"])

                # Get discriminator predictions for generator training
                with torch.no_grad():
                    disc_fake = self.model_disc(batch["gen_output"])

                # Calculate generator losses
                batch["gen_loss"] = self.criterion_gen(disc_fake, batch)
                del disc_fake
                torch.cuda.empty_cache()

                # Scale and backward
                scaled_gen_loss = batch["gen_loss"] / self.gradient_accumulation_steps
                scaled_gen_loss.backward()
                del scaled_gen_loss

                self._clip_grad_norm(self.model_gen)

                if (self.current_accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer_gen.step()

                self.current_accumulation_step = (self.current_accumulation_step + 1) % self.gradient_accumulation_steps

            else:
                # Inference mode
                with torch.no_grad():
                    batch["lr_image"] = self.degrader.process_batch(batch["data_object"])
                    batch["gen_output"] = self.model_gen(batch["lr_image"])

                    disc_fake = self.model_disc(batch["gen_output"])
                    disc_real = self.model_disc(batch["data_object"])

                    batch["disc_loss"] = self.criterion_disc(disc_fake, disc_real, batch)
                    batch["gen_loss"] = self.criterion_gen(disc_fake, batch)

                    del disc_fake, disc_real
                    torch.cuda.empty_cache()

            # Update metrics only for logging step
            if self.is_train and (self.current_accumulation_step + 1) % self.log_step == 0:
                # Store only the metrics we need for logging
                self.last_metrics = {}
                for met in metric_funcs:
                    self.last_metrics[met.name] = met(**batch)
                # Store current losses
                self.last_metrics["gen_loss"] = batch["gen_loss"].item()
                self.last_metrics["disc_loss"] = batch["disc_loss"].item()
                # Update MetricTracker only with the latest values
                for name, value in self.last_metrics.items():
                    metrics.update(name, value)
            elif not self.is_train:
                # During inference, update all metrics
                for met in metric_funcs:
                    metrics.update(met.name, met(**batch))

            return batch

        except Exception as e:
            torch.cuda.empty_cache()
            raise e

    def _log_batch(self, batch_idx, batch, mode="train"):
        if mode == "train":
            if batch_idx % self.log_step == 0:
                # Log only single image for visualization
                with torch.no_grad():
                    self.writer.add_images("train/real", batch["data_object"][:1].detach())
                    self.writer.add_images("train/generated", batch["gen_output"][:1].detach())
                    self.writer.add_images("train/low_res", batch["lr_image"][:1].detach())
                # Clear references to ensure memory is freed
                torch.cuda.empty_cache()
        else:
            # During validation, also log only single image
            with torch.no_grad():
                self.writer.add_images("val/real", batch["data_object"][:1].detach())
                self.writer.add_images("val/generated", batch["gen_output"][:1].detach())
                self.writer.add_images("val/low_res", batch["lr_image"][:1].detach())
            torch.cuda.empty_cache()