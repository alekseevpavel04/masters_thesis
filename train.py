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

    # setup data_loader instances
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture based on model_type
    if config.trainer.model_type == "GAN":
        model_gen = instantiate(config.model_gen).to(device)
        model_disc = instantiate(config.model_disc).to(device)
        n_params_gen = count_parameters(model_gen)
        n_params_disc = count_parameters(model_disc)
        logger.info(f"Generator training parameters: {n_params_gen:,}")
        logger.info(f"Discriminator training parameters: {n_params_disc:,}")
        logger.info(f"Total training parameters: {n_params_gen + n_params_disc:,}")
    elif config.trainer.model_type == "regular":
        model_diff = instantiate(config.model_diff).to(device)
        n_params_diff = count_parameters(model_diff)
        logger.info(f"Model training parameters: {n_params_diff:,}")
    else:
        raise ValueError(f"Unknown model type: {config.trainer.model_type}")

    # get function handles of loss and metrics
    loss_function_gen = instantiate(config.loss_function_gen).to(device)
    loss_function_disc = instantiate(config.loss_function_disc).to(device)
    loss_function_diff = instantiate(config.loss_function_diff).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    if config.trainer.model_type == "GAN":
        trainable_params_gen = filter(lambda p: p.requires_grad, model_gen.parameters())
        optimizer_gen = instantiate(config.optimizer_gen, params=trainable_params_gen)
        lr_scheduler_gen = instantiate(config.lr_scheduler_gen, optimizer=optimizer_gen)

        trainable_params_disc = filter(lambda p: p.requires_grad, model_disc.parameters())
        optimizer_disc = instantiate(config.optimizer_disc, params=trainable_params_disc)
        lr_scheduler_disc = instantiate(config.lr_scheduler_disc, optimizer=optimizer_disc)

    elif config.trainer.model_type == "regular":
        trainable_params_diff = filter(lambda p: p.requires_grad, model_diff.parameters())
        optimizer_diff = instantiate(config.optimizer_diff, params=trainable_params_diff)
        lr_scheduler_diff = instantiate(config.lr_scheduler_diff, optimizer=optimizer_diff)

    degrader = instantiate(config.degradation)

    epoch_len = config.trainer.get("epoch_len")
    disc_steps = config.trainer.get("disc_steps", 1)  # для диффузии это не нужно
    gradient_accumulation_steps = config.trainer.get("gradient_accumulation_steps", 1)

    trainer = Trainer(
        model_gen=model_gen if config.trainer.model_type == "GAN" else None,
        model_disc=model_disc if config.trainer.model_type == "GAN" else None,
        model_diff=model_diff if config.trainer.model_type == "regular" else None,
        criterion_gen=loss_function_gen if config.trainer.model_type == "GAN" else None,
        criterion_disc=loss_function_disc if config.trainer.model_type == "GAN" else None,
        criterion_diff=loss_function_diff if config.trainer.model_type == "regular" else None,
        metrics=metrics,
        optimizer_gen=optimizer_gen if config.trainer.model_type == "GAN" else None,
        optimizer_disc=optimizer_disc if config.trainer.model_type == "GAN" else None,
        optimizer_diff=optimizer_diff if config.trainer.model_type == "regular" else None,
        lr_scheduler_gen=lr_scheduler_gen if config.trainer.model_type == "GAN" else None,
        lr_scheduler_disc=lr_scheduler_disc if config.trainer.model_type == "GAN" else None,
        lr_scheduler_diff=lr_scheduler_diff if config.trainer.model_type == "regular" else None,
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