import warnings
import hydra
import torch
from hydra.utils import instantiate
from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.io_utils import ROOT_PATH

from omegaconf import OmegaConf

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for GAN inference. Instantiates the generator model, metrics,
    and dataloaders. Runs Inferencer to generate high-resolution images and
    calculate metrics.

    Args:
        config (DictConfig): hydra experiment config.
    """

    project_config = OmegaConf.to_container(config)
    set_random_seed(config.inferencer.seed)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances and batch_transforms
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build generator model only (we don't need discriminator for inference)
    model_gen = instantiate(config.model_gen).to(device)

    # get metrics for super-resolution evaluation
    metrics = instantiate(config.metrics)

    log_step = config.inferencer.log_step

    degrader = instantiate(config.degradation)

    # save_path for generated high-resolution images
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model_gen=model_gen,
        degrader=degrader,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        writer=writer,
        logger=logger,
        log_step=log_step,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    # Print metrics for each partition
    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")

if __name__ == "__main__":
    main()