from utils import get_best_device
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="pipeline")
def configure(cfg: DictConfig) -> DictConfig:
    cfg.device = get_best_device()
    return cfg


config = configure()
