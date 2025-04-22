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

                if self.config.trainer.model_type == "GAN":
                    # GAN training logic
                    for _ in range(self.disc_steps):
                        if self.current_accumulation_step == 0:
                            self.optimizer_disc.zero_grad()

                        batch["lr_image"] = self.degrader.process_batch(batch["data_object"])

                        with torch.no_grad():
                            batch["gen_output"] = self.model_gen(batch["lr_image"])

                        disc_fake = self.model_disc(batch["gen_output"].detach())
                        disc_real = self.model_disc(batch["data_object"])

                        batch["disc_loss"] = self.criterion_disc(disc_fake, disc_real, batch)

                        del disc_fake, disc_real
                        torch.cuda.empty_cache()

                        scaled_disc_loss = batch["disc_loss"] / self.gradient_accumulation_steps
                        scaled_disc_loss.backward()
                        del scaled_disc_loss

                        self._clip_grad_norm(self.model_disc)

                        if (self.current_accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                            self.optimizer_disc.step()

                    if self.current_accumulation_step == 0:
                        self.optimizer_gen.zero_grad()

                    batch["lr_image"] = self.degrader.process_batch(batch["data_object"])

                    batch["gen_output"] = self.model_gen(batch["lr_image"])

                    with torch.no_grad():
                        disc_fake = self.model_disc(batch["gen_output"])

                    batch["gen_loss"] = self.criterion_gen(disc_fake, batch)
                    del disc_fake
                    torch.cuda.empty_cache()

                    scaled_gen_loss = batch["gen_loss"] / self.gradient_accumulation_steps
                    scaled_gen_loss.backward()
                    del scaled_gen_loss

                    self._clip_grad_norm(self.model_gen)

                    if (self.current_accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer_gen.step()

                elif self.config.trainer.model_type == "regular":
                    # Regular model training logic
                    if self.current_accumulation_step == 0:
                        self.optimizer_diff.zero_grad()

                    batch["lr_image"] = self.degrader.process_batch(batch["data_object"])

                    # Check for custom model type
                    custom_type = getattr(self.config.trainer, "custom_type", None)

                    if custom_type == "Swin2SR":
                        # Handle Swin2SR model output
                        output = self.model_diff(batch["lr_image"])

                        # If it's a tuple, take the first element
                        if isinstance(output, tuple):
                            batch["diff_output"] = output[0]
                        else:
                            batch["diff_output"] = output
                    else:
                        # Standard handling for other models
                        batch["diff_output"] = self.model_diff(batch["lr_image"])

                    batch["diff_loss"] = self.criterion_diff(batch["diff_output"], batch["data_object"])

                    scaled_diff_loss = batch["diff_loss"] / self.gradient_accumulation_steps
                    scaled_diff_loss.backward()
                    del scaled_diff_loss

                    self._clip_grad_norm(self.model_diff)

                    if (self.current_accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer_diff.step()

                    self.current_accumulation_step = (
                                                                 self.current_accumulation_step + 1) % self.gradient_accumulation_steps

            # For validation/inference or metric calculation during training
            else:
                with torch.no_grad():
                    # If processes are skipped during validation, we need to ensure lr_image, gen_output/diff_output exist
                    if "lr_image" not in batch:
                        batch["lr_image"] = self.degrader.process_batch(batch["data_object"])

                    if self.config.trainer.model_type == "GAN":
                        if "gen_output" not in batch:
                            batch["gen_output"] = self.model_gen(batch["lr_image"])

                        if "disc_loss" not in batch or "gen_loss" not in batch:
                            disc_fake = self.model_disc(batch["gen_output"])
                            disc_real = self.model_disc(batch["data_object"])

                            batch["disc_loss"] = self.criterion_disc(disc_fake, disc_real, batch)
                            batch["gen_loss"] = self.criterion_gen(disc_fake, batch)

                            del disc_fake, disc_real
                    elif self.config.trainer.model_type == "regular":
                        # Check for custom model type
                        custom_type = getattr(self.config.trainer, "custom_type", None)

                        if "diff_output" not in batch:
                            if custom_type == "Swin2SR":
                                # Handle Swin2SR model output
                                output = self.model_diff(batch["lr_image"])

                                # If it's a tuple, take the first element
                                if isinstance(output, tuple):
                                    batch["diff_output"] = output[0]
                                else:
                                    batch["diff_output"] = output
                            else:
                                # Standard handling for other models
                                batch["diff_output"] = self.model_diff(batch["lr_image"])

                        if "diff_loss" not in batch:
                            batch["diff_loss"] = self.criterion_diff(batch["diff_output"], batch["data_object"])

                    torch.cuda.empty_cache()

            # Update metrics during training at specified intervals
            if self.is_train and (self.current_accumulation_step + 1) % self.log_step == 0:
                self.last_metrics = {}
                for met in metric_funcs:
                    metric_value = met(**batch)
                    self.last_metrics[met.name] = metric_value
                if self.config.trainer.model_type == "GAN":
                    self.last_metrics["gen_loss"] = batch["gen_loss"].item()
                    self.last_metrics["disc_loss"] = batch["disc_loss"].item()
                elif self.config.trainer.model_type == "regular":
                    self.last_metrics["diff_loss"] = batch["diff_loss"].item()
                for name, value in self.last_metrics.items():
                    metrics.update(name, value)
            # Update metrics during validation/inference
            elif not self.is_train:
                for met in metric_funcs:
                    try:
                        metric_value = met(**batch)
                        metrics.update(met.name, metric_value)
                    except Exception as e:
                        self.logger.warning(f"Error calculating metric {met.name}: {str(e)}")
                        # Continue with other metrics even if one fails

            return batch

        except Exception as e:
            torch.cuda.empty_cache()
            raise e

    def _log_batch(self, batch_idx, batch, mode="train"):
        if mode == "train":
            # log images 10 times more rare than log_step
            if batch_idx % (self.log_step * 10) == 0:
                with torch.no_grad():
                    self.writer.add_images(f"{mode}/real", torch.clamp(batch["data_object"][:1].detach(), 0, 1))
                    self.writer.add_images(f"{mode}/low_res", torch.clamp(batch["lr_image"][:1].detach(), 0, 1))

                    if self.config.trainer.model_type == "GAN":
                        self.writer.add_images(f"{mode}/generated", torch.clamp(batch["gen_output"][:1].detach(), 0, 1))
                    elif self.config.trainer.model_type == "regular":
                        self.writer.add_images(f"{mode}/generated",
                                               torch.clamp(batch["diff_output"][:1].detach(), 0, 1))

                torch.cuda.empty_cache()
        else:
            # During validation, also log only single image
            with torch.no_grad():
                self.writer.add_images("val/real", torch.clamp(batch["data_object"][:1].detach(), 0, 1))
                self.writer.add_images("val/low_res", torch.clamp(batch["lr_image"][:1].detach(), 0, 1))
                if self.config.trainer.model_type == "GAN":
                    self.writer.add_images(f"val/generated", torch.clamp(batch["gen_output"][:1].detach(), 0, 1))
                elif self.config.trainer.model_type == "regular":
                    self.writer.add_images(f"val/generated", torch.clamp(batch["diff_output"][:1].detach(), 0, 1))
            torch.cuda.empty_cache()