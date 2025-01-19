import warnings
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging


def count_parameters(model):
    """
    Count number of trainable parameters in the model

    Args:
        model: PyTorch model
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    print(device)
    # setup data_loader instances
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model_gen = instantiate(config.model_gen).to(device)
    model_disc = instantiate(config.model_disc).to(device)

    # Count and log number of parameters
    n_params_gen = count_parameters(model_gen)
    n_params_disc = count_parameters(model_disc)

    logger.info(f"Generator training parameters: {n_params_gen:,}")
    logger.info(f"Discriminator training parameters: {n_params_disc:,}")
    logger.info(f"Total training parameters: {n_params_gen + n_params_disc:,}")

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

    degrader = instantiate(config.degradation)

    epoch_len = config.trainer.get("epoch_len")
    disc_steps = config.trainer.get("disc_steps")
    gradient_accumulation_steps = config.trainer.get("gradient_accumulation_steps")

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
        degrader=degrader,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        disc_steps=disc_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()