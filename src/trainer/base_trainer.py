from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model_gen,
        model_disc,
        criterion_gen,
        criterion_disc,
        metrics,
        optimizer_gen,
        optimizer_disc,
        lr_scheduler_gen,
        lr_scheduler_disc,
        config,
        device,
        dataloaders,
        logger,
        writer,
        disc_steps = 5,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model_gen = model_gen
        self.model_disc = model_disc
        self.criterion_gen = criterion_gen
        self.criterion_disc = criterion_disc
        self.optimizer_gen = optimizer_gen
        self.optimizer_disc = optimizer_disc
        self.lr_scheduler_gen = lr_scheduler_gen
        self.lr_scheduler_disc = lr_scheduler_disc

        self.batch_transforms = batch_transforms
        self.disc_steps = disc_steps

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # configuration to monitor model performance and save best

        self.save_period = (
            self.cfg_trainer.save_period
        )  # checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        )  # format: "mnt_mode mnt_metric"

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

        # setup visualization writer instance
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

        # define checkpoint dir and init everything if required

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)
        print(config.trainer.get("from_pretrained_gen"))
        print(config.trainer.get("from_pretrained_disc"))
        if config.trainer.get("from_pretrained_gen") is not None and config.trainer.get(
                "from_pretrained_disc") is not None:
            self._from_pretrained(
                config.trainer.get("from_pretrained_gen"),
                config.trainer.get("from_pretrained_disc")
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
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.model_gen.train()
        self.model_disc.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)

        # Initialize variable for storing the last batch's metrics
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
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            # Get gradient norms for both generator and discriminator
            grad_norm_gen, grad_norm_disc = self._get_grad_norm()

            # Update progress bar description with metrics while keeping the progress bar
            pbar.set_description(
                f"train | loss: {batch['loss'].item():.3f} | g_loss: {batch['gen_loss'].item():.3f} | d_loss: {batch['disc_loss'].item():.3f} | g_n_gen: {grad_norm_gen:.3f} | g_n_disc: {grad_norm_disc:.3f}"
            )

            # Log gradient norms only for specific logging
            self.train_metrics.update("grad_norm_gen", grad_norm_gen)
            self.train_metrics.update("grad_norm_disc", grad_norm_disc)

            self.train_metrics.update("gen_loss", batch["gen_loss"])
            self.train_metrics.update("disc_loss", batch["disc_loss"])

            # Log current results periodically
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar("learning rate gen", self.lr_scheduler_gen.get_last_lr()[0])
                self.writer.add_scalar("learning rate disc", self.lr_scheduler_disc.get_last_lr()[0])
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

            if batch_idx + 1 >= self.epoch_len:
                break

        logs = last_train_metrics

        # Run val/test
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

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
        self.model_gen.eval()
        self.model_disc.eval()
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

        return self.evaluation_metrics.result()

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
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch_gen = type(self.model_gen).__name__
        arch_disc = type(self.model_disc).__name__

        state = {
            "arch_gen": arch_gen,
            "arch_disc": arch_disc,
            "epoch": epoch,
            "state_dict_gen": self.model_gen.state_dict(),
            "state_dict_disc": self.model_disc.state_dict(),
            "optimizer_gen": self.optimizer_gen.state_dict(),
            "optimizer_disc": self.optimizer_disc.state_dict(),
            "lr_scheduler_gen": self.lr_scheduler_gen.state_dict(),
            "lr_scheduler_disc": self.lr_scheduler_disc.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }

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

    def _from_pretrained(self, pretrained_path_gen, pretrained_path_disc=None, mode="Train"):
        """
        Init models with weights from pretrained files.

        Args:
            pretrained_path_gen (str): path to the generator model state dict.
            pretrained_path_disc (str | None): path to the discriminator model state dict.
                Required for Train mode, ignored for Inference mode.
            mode (str): "Train" or "Inference". In Inference mode, only generator
                weights are loaded.
        """
        pretrained_path_gen = str(pretrained_path_gen)

        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading generator weights from: {pretrained_path_gen} ...")
            if mode == "Train" and pretrained_path_disc is not None:
                self.logger.info(f"Loading discriminator weights from: {pretrained_path_disc} ...")
        else:
            print(f"Loading generator weights from: {pretrained_path_gen} ...")
            if mode == "Train" and pretrained_path_disc is not None:
                print(f"Loading discriminator weights from: {pretrained_path_disc} ...")

        # Load generator checkpoint
        checkpoint_gen = torch.load(pretrained_path_gen, map_location=self.device, weights_only=True)
        if "state_dict_gen" in checkpoint_gen:
            self.model_gen.load_state_dict(checkpoint_gen["state_dict_gen"])
        elif "params_ema" in checkpoint_gen:
            self.model_gen.load_state_dict(checkpoint_gen["params_ema"])
        else:
            self.model_gen.load_state_dict(checkpoint_gen)  # Assuming it's a direct state dict

        # Load discriminator checkpoint only in Train mode
        if mode == "Train" and pretrained_path_disc is not None:
            pretrained_path_disc = str(pretrained_path_disc)
            checkpoint_disc = torch.load(pretrained_path_disc, map_location=self.device, weights_only=True)
            if "state_dict_disc" in checkpoint_disc:
                self.model_disc.load_state_dict(checkpoint_disc["state_dict_disc"])
            elif "params" in checkpoint_disc:
                self.model_disc.load_state_dict(checkpoint_disc["params"])
            else:
                self.model_disc.load_state_dict(checkpoint_disc)  # Assuming it's a direct state dict
