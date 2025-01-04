import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model_gen = instantiate(config.model_gen).to(device)
    # logger.info(model_gen)
    model_disc = instantiate(config.model_disc).to(device)
    # logger.info(model_disc)
    # get function handles of loss and metrics
    loss_function_gen = instantiate(config.loss_function_gen).to(device)
    loss_function_disc = instantiate(config.loss_function_disc).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params_gen = filter(lambda p: p.requires_grad, model_gen.parameters())
    optimizer_gen = instantiate(config.optimizer_gen, params=trainable_params_gen)
    lr_scheduler_gen = instantiate(config.lr_scheduler_gen, optimizer=optimizer_gen)

    trainable_params_disc = filter(lambda p: p.requires_grad, model_gen.parameters())
    optimizer_disc = instantiate(config.optimizer_disc, params=trainable_params_disc)
    lr_scheduler_disc = instantiate(config.lr_scheduler_disc, optimizer=optimizer_disc)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")
    disc_steps = config.trainer.get("disc_steps")

    trainer = Trainer(
        model_gen=model_gen,
        model_disc=model_disc,
        criterion_gen=loss_function_gen,
        criterion_disc=loss_function_disc,
        metrics=metrics,
        optimizer_gen=optimizer_gen,
        optimizer_disc=optimizer_disc,
        lr_scheduler_gen=lr_scheduler_gen,
        lr_scheduler_disc=lr_scheduler_disc,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        disc_steps = disc_steps,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()