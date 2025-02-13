from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH

from collections import OrderedDict


class BaseTrainer:
    def __init__(
            self,
            model_gen=None,
            model_disc=None,
            model_diff=None,
            criterion_gen=None,
            criterion_disc=None,
            criterion_diff=None,
            metrics=None,
            optimizer_gen=None,
            optimizer_disc=None,
            optimizer_diff=None,
            lr_scheduler_gen=None,
            lr_scheduler_disc=None,
            lr_scheduler_diff=None,
            degrader=None,
            config=None,
            device=None,
            dataloaders=None,
            logger=None,
            writer=None,
            disc_steps=None,
            gradient_accumulation_steps=None,
            epoch_len=None,
            skip_oom=True,
            batch_transforms=None,
    ):
        self.is_train = True
        self.config = config
        self.cfg_trainer = self.config.trainer
        self.model_type = self.cfg_trainer.model_type

        self.device = device
        self.skip_oom = skip_oom
        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        # Initialize models based on model type
        if self.model_type == "GAN":
            self.model_gen = model_gen
            self.model_disc = model_disc
            self.criterion_gen = criterion_gen
            self.criterion_disc = criterion_disc
            self.optimizer_gen = optimizer_gen
            self.optimizer_disc = optimizer_disc
            self.lr_scheduler_gen = lr_scheduler_gen
            self.lr_scheduler_disc = lr_scheduler_disc
        elif self.model_type == "regular":
            self.model_diff = model_diff
            self.criterion_diff = criterion_diff
            self.optimizer_diff = optimizer_diff
            self.lr_scheduler_diff = lr_scheduler_diff
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.degrader = degrader
        self.batch_transforms = batch_transforms
        self.disc_steps = disc_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            self.epoch_len = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # configuration to monitor model performance and save best
        self.save_period = self.cfg_trainer.save_period
        self.monitor = self.cfg_trainer.get("monitor", "off")

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]
            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        self.checkpoint_dir = ROOT_PATH / config.trainer.save_dir / config.writer.run_name

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        # Load pretrained weights if specified
        if (config.trainer.get("from_pretrained_gen") is not None
                or config.trainer.get("from_pretrained_disc") is not None
                or config.trainer.get("from_pretrained_diff") is not None):
            self._from_pretrained(
                config.trainer.get("from_pretrained_gen"),
                config.trainer.get("from_pretrained_disc"),
                config.trainer.get("from_pretrained_diff")
            )

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

            if stop_process:  # early_stop
                break

    def _train_epoch(self, epoch):
        self.is_train = True

        # Set models to train mode based on model type
        if self.model_type == "GAN":
            self.model_gen.train()
            self.model_disc.train()
        else:  # regular
            self.model_diff.train()

        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)

        last_train_metrics = {}

        pbar = tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        for batch_idx, batch in enumerate(pbar):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            # Update progress bar description based on model type
            if self.model_type == "GAN":
                grad_norm_gen, grad_norm_disc = self._get_grad_norm()
                pbar.set_description(
                    f"train | g_loss: {batch['gen_loss'].item():.3f} | d_loss: {batch['disc_loss'].item():.3f} | "
                    f"g_n_gen: {grad_norm_gen:.3f} | g_n_disc: {grad_norm_disc:.3f}"
                )
                self.train_metrics.update("grad_norm_gen", grad_norm_gen)
                self.train_metrics.update("grad_norm_disc", grad_norm_disc)
                self.train_metrics.update("gen_loss", batch["gen_loss"])
                self.train_metrics.update("disc_loss", batch["disc_loss"])
            else:  # regular
                grad_norm = self._get_grad_norm_diff()
                pbar.set_description(
                    f"train | diff_loss: {batch['diff_loss'].item():.3f} | g_norm: {grad_norm:.3f}"
                )
                self.train_metrics.update("grad_norm", grad_norm)
                self.train_metrics.update("diff_loss", batch["diff_loss"])

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch,
                        self._progress(batch_idx),
                        batch["diff_loss" if self.model_type == "regular" else "gen_loss"].item(),
                    )
                )
                # Log learning rate based on model type
                if self.model_type == "GAN":
                    self.writer.add_scalar("learning rate gen", self.lr_scheduler_gen.get_last_lr()[0])
                    self.writer.add_scalar("learning rate disc", self.lr_scheduler_disc.get_last_lr()[0])
                else:  # regular
                    self.writer.add_scalar("learning rate", self.lr_scheduler_diff.get_last_lr()[0])

                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

            if batch_idx + 1 >= self.epoch_len:
                break

        logs = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs


    @torch.no_grad()
    def _get_grad_norm_diff(self, norm_type=2):
        """Calculate gradient norm for regular model"""
        parameters = self.model_diff.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False

        # Adjusting model evaluation based on model type
        if self.model_type == "GAN":
            self.model_gen.eval()
            self.model_disc.eval()
        elif self.model_type == "regular":
            self.model_diff.eval()

        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(
                batch_idx, batch, part
            )  # log only the last batch during inference

        result = self.evaluation_metrics.result()

        # Remove loss terms for evaluation
        loss_keys = ['gen_loss', 'disc_loss', 'diff_loss']
        for key in loss_keys:
            if key in result:
                del result[key]

        return result

    def _monitor_performance(self, logs, not_improved_count):
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
                if self.mnt_mode == "min":
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == "max":
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                stop_process = True
        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        # do batch transforms on device
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self, model):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging, now for both the generator and the discriminator.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm_gen (float), total_norm_disc (float): the calculated norm for generator and discriminator.
        """
        # Calculate gradient norm for generator
        gen_parameters = self.model_gen.parameters()
        if isinstance(gen_parameters, torch.Tensor):
            gen_parameters = [gen_parameters]
        gen_parameters = [p for p in gen_parameters if p.grad is not None]
        total_norm_gen = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in gen_parameters]),
            norm_type,
        )

        # Calculate gradient norm for discriminator
        disc_parameters = self.model_disc.parameters()
        if isinstance(disc_parameters, torch.Tensor):
            disc_parameters = [disc_parameters]
        disc_parameters = [p for p in disc_parameters if p.grad is not None]
        total_norm_disc = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in disc_parameters]),
            norm_type,
        )

        return total_norm_gen.item(), total_norm_disc.item()

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """Modified to handle both GAN and regular model checkpoints"""
        if self.model_type == "GAN":
            state = {
                "arch_gen": type(self.model_gen).__name__,
                "arch_disc": type(self.model_disc).__name__,
                "epoch": epoch,
                "state_dict_gen": self.model_gen.state_dict(),
                "state_dict_disc": self.model_disc.state_dict(),
                "optimizer_gen": self.optimizer_gen.state_dict(),
                "optimizer_disc": self.optimizer_disc.state_dict(),
                "lr_scheduler_gen": self.lr_scheduler_gen.state_dict(),
                "lr_scheduler_disc": self.lr_scheduler_disc.state_dict(),
            }
        else:  # regular
            state = {
                "arch_diff": type(self.model_diff).__name__,
                "epoch": epoch,
                "state_dict_diff": self.model_diff.state_dict(),
                "optimizer_diff": self.optimizer_diff.state_dict(),
                "lr_scheduler_diff": self.lr_scheduler_diff.state_dict(),
            }

        state.update({
            "monitor_best": self.mnt_best,
            "config": self.config,
        })

        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")

        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, map_location=self.device)

        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # Load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )

        # Load the state dicts for both the generator and discriminator
        self.model_gen.load_state_dict(checkpoint["state_dict_gen"])
        self.model_disc.load_state_dict(checkpoint["state_dict_disc"])

        # Load optimizer and scheduler states for both generator and discriminator
        if checkpoint["config"]["optimizer"] != self.config["optimizer"] or checkpoint["config"]["lr_scheduler"] != \
                self.config["lr_scheduler"]:
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizer_gen.load_state_dict(checkpoint["optimizer_gen"])
            self.optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
            self.lr_scheduler_gen.load_state_dict(checkpoint["lr_scheduler_gen"])
            self.lr_scheduler_disc.load_state_dict(checkpoint["lr_scheduler_disc"])

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")

    def _from_pretrained(self, pretrained_path_gen=None, pretrained_path_disc=None, pretrained_path_diff=None,
                         mode="Train"):
        """
        Init models with weights from pretrained files.

        Args:
            pretrained_path_gen (str): path to the generator model state dict.
            pretrained_path_disc (str | None): path to the discriminator model state dict.
                Required for Train mode, ignored for Inference mode.
            pretrained_path_diff (str | None): path to the regular model state dict.
            mode (str): "Train" or "Inference". In Inference mode, only generator
                weights are loaded.
        """
        if self.config.trainer.model_type == "GAN":
            if pretrained_path_gen is not None:
                self._load_pretrained_gen(self.model_gen, pretrained_path_gen, "generator")
            if mode == "Train" and pretrained_path_disc is not None:
                self._load_pretrained_disc(self.model_disc, pretrained_path_disc, "discriminator")
        elif self.config.trainer.model_type == "regular":
            if pretrained_path_diff is not None:
                self._load_pretrained_diff(self.model_diff, pretrained_path_diff, "regular")
        else:
            raise ValueError(f"Unknown model type: {self.config.trainer.model_type}")

    def _load_pretrained_gen(self, model, pretrained_path, model_name):
        """
        Helper method to load pretrained weights into a model.

        Args:
            model (nn.Module): The model to load weights into.
            pretrained_path (str): Path to the pretrained weights.
            model_name (str): Name of the model (for logging purposes).
        """
        if hasattr(self, "logger"):
            self.logger.info(f"Loading {model_name} weights from: {pretrained_path} ...")
        else:
            print(f"Loading {model_name} weights from: {pretrained_path} ...")

        checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "params_ema" in checkpoint:
            model.load_state_dict(checkpoint["params_ema"])
        else:
            model.load_state_dict(checkpoint)  # Assuming it's a direct state dict
    def _load_pretrained_disc(self, model, pretrained_path, model_name):
        if hasattr(self, "logger"):
           self.logger.info(f"Loading {model_name} weights from: {pretrained_path} ...")
        else:
           print(f"Loading {model_name} weights from: {pretrained_path} ...")

        pretrained_path_disc = str(pretrained_path)
        checkpoint_disc = torch.load(pretrained_path_disc, map_location=self.device, weights_only=True)
        if "state_dict_disc" in checkpoint_disc:
           model.load_state_dict(checkpoint_disc["state_dict_disc"])
        elif "params" in checkpoint_disc:
           model.load_state_dict(checkpoint_disc["params"])
        else:
           model.load_state_dict(checkpoint_disc)  # Assuming it's a direct state dict

    def _load_pretrained_diff(self, model, pretrained_path, model_name):
        if hasattr(self, "logger"):
            self.logger.info(f"Loading {model_name} weights from: {pretrained_path} ...")
        else:
            print(f"Loading {model_name} weights from: {pretrained_path} ...")

        checkpoint = torch.load(pretrained_path, map_location=self.device)

        # Get state_dict with correct key
        if 'state_dict_diff' in checkpoint:
            state_dict = checkpoint['state_dict_diff']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'params' in checkpoint:  # Add handling for 'params' key
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint

        model_dict = model.state_dict()

        # Find mismatched keys
        unexpected_keys = [k for k in state_dict.keys() if k not in model_dict]
        missing_keys = [k for k in model_dict.keys() if k not in state_dict]

        if unexpected_keys:
            print(f"Warning: Found unexpected keys in checkpoint: {unexpected_keys}")
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")

        # Filter state_dict
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

        # Try loading with strict=True first
        try:
            model.load_state_dict(filtered_state_dict, strict=True)
            print("Weights loaded successfully with strict=True.")
        except Exception as e:
            print(f"Warning: Strict loading failed: {str(e)}")
            try:
                # If strict loading fails, try with strict=False
                model.load_state_dict(filtered_state_dict, strict=False)
                print("Weights loaded successfully with strict=False.")
            except Exception as e:
                print(f"Error: Failed to load weights even with strict=False: {str(e)}")
                raise e  # Re-raise the exception if both attempts fail

        return model