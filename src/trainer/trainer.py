from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.degradation import ImageDegrader
import torch


class Trainer(BaseTrainer):
    def __init__(self, *args, gradient_accumulation_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.degrader = ImageDegrader(mode='batch', device=self.device)
        self.content_weight = 1.0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.current_accumulation_step = 0

    def _reset_accumulation_step(self):
        """Reset gradient accumulation counter"""
        self.current_accumulation_step = 0

    def _train_epoch(self, epoch):
        self._reset_accumulation_step()
        try:
            result = super()._train_epoch(epoch)
            torch.cuda.empty_cache()  # Очистка в конце эпохи
            return result
        except Exception as e:
            torch.cuda.empty_cache()  # Очистка при ошибке
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
                    disc_loss_fake = self.criterion_disc(disc_fake, False)
                    disc_loss_real = self.criterion_disc(disc_real, True)
                    batch["disc_loss"] = (disc_loss_fake + disc_loss_real) * 0.5

                    # Explicitly delete intermediate tensors
                    del disc_fake, disc_real, disc_loss_fake, disc_loss_real
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
                batch["gen_loss"] = self.criterion_gen(disc_fake, True)
                del disc_fake
                torch.cuda.empty_cache()

                # Add content loss
                content_loss = torch.nn.functional.l1_loss(batch["gen_output"], batch["data_object"])
                batch["gen_loss"] += self.content_weight * content_loss
                del content_loss

                # Scale and backward
                scaled_gen_loss = batch["gen_loss"] / self.gradient_accumulation_steps
                scaled_gen_loss.backward()
                del scaled_gen_loss

                self._clip_grad_norm(self.model_gen)

                if (self.current_accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer_gen.step()

                self.current_accumulation_step = (self.current_accumulation_step + 1) % self.gradient_accumulation_steps

                # Combine losses for logging
                batch["loss"] = batch["gen_loss"] + batch["disc_loss"]

            else:
                # Inference mode
                with torch.no_grad():
                    batch["lr_image"] = self.degrader.process_batch(batch["data_object"])
                    batch["gen_output"] = self.model_gen(batch["lr_image"])

                    disc_fake = self.model_disc(batch["gen_output"])
                    disc_real = self.model_disc(batch["data_object"])

                    batch["disc_loss"] = self.criterion_disc(disc_fake, False) + self.criterion_disc(disc_real, True)
                    batch["gen_loss"] = self.criterion_gen(disc_fake, True)

                    del disc_fake, disc_real
                    torch.cuda.empty_cache()

                    content_loss = torch.nn.functional.l1_loss(batch["gen_output"], batch["data_object"])
                    batch["gen_loss"] += self.content_weight * content_loss
                    del content_loss

                    batch["loss"] = batch["gen_loss"] + batch["disc_loss"]

            # Update metrics
            for met in metric_funcs:
                metrics.update(met.name, met(**batch))

            return batch

        except Exception as e:
            # Clean up on error
            torch.cuda.empty_cache()
            raise e

    def _log_batch(self, batch_idx, batch, mode="train"):
        if mode == "train":
            if batch_idx % self.log_step == 0:
                self.writer.add_images("train/real", batch["data_object"][0:1])
                self.writer.add_images("train/generated", batch["gen_output"][0:1])
                self.writer.add_images("train/low_res", batch["lr_image"][0:1])
        else:
            self.writer.add_images("val/real", batch["data_object"][0:1])
            self.writer.add_images("val/generated", batch["gen_output"][0:1])
            self.writer.add_images("val/low_res", batch["lr_image"][0:1])