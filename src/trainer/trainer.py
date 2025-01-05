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
        # Reset accumulation step at the start of each epoch
        self._reset_accumulation_step()
        return super()._train_epoch(epoch)

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

            # Train Discriminator
            for _ in range(self.disc_steps):
                # Zero gradients only at the start of accumulation
                if self.current_accumulation_step == 0:
                    self.optimizer_disc.zero_grad()

                # Generate low resolution images
                lr_image = self.degrader.process_batch(batch["data_object"])
                batch["lr_image"] = lr_image

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

                # Scale loss by accumulation steps
                scaled_disc_loss = batch["disc_loss"] / self.gradient_accumulation_steps
                scaled_disc_loss.backward()

                # Clip gradients after each backward pass
                self._clip_grad_norm(self.model_disc)

                # Only update weights after accumulating enough gradients
                if (self.current_accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer_disc.step()

            # Train Generator
            # Zero gradients only at the start of accumulation
            if self.current_accumulation_step == 0:
                self.optimizer_gen.zero_grad()

            # Generate new fake samples
            lr_image = self.degrader.process_batch(batch["data_object"])
            batch["lr_image"] = lr_image

            # Generate fake images
            batch["gen_output"] = self.model_gen(batch["lr_image"])

            # Get discriminator predictions for generator training
            with torch.no_grad():
                disc_fake = self.model_disc(batch["gen_output"])

            # Calculate generator losses
            batch["gen_loss"] = self.criterion_gen(disc_fake, True)

            # Add content loss
            content_loss = torch.nn.functional.l1_loss(batch["gen_output"], batch["data_object"])
            batch["gen_loss"] += self.content_weight * content_loss

            # Scale loss by accumulation steps
            scaled_gen_loss = batch["gen_loss"] / self.gradient_accumulation_steps
            scaled_gen_loss.backward()

            # Clip gradients after each backward pass
            self._clip_grad_norm(self.model_gen)

            # Only update weights after accumulating enough gradients
            if (self.current_accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer_gen.step()

            # Update accumulation step counter
            self.current_accumulation_step = (self.current_accumulation_step + 1) % self.gradient_accumulation_steps

            # Combine losses for logging (use unscaled losses)
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

                content_loss = torch.nn.functional.l1_loss(batch["gen_output"], batch["data_object"])
                batch["gen_loss"] += self.content_weight * content_loss

                batch["loss"] = batch["gen_loss"] + batch["disc_loss"]

        # Update metrics
        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch

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