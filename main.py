from config import config
from utils import logger, get_best_device
from data_pipeline import DataPipeline
from rich.console import Console


def main():
    logger.info(f'Keypoint Data Pipeline')
    logger.info(f'Task      : {config.task}')
    logger.info(f'Detector  : {config.components.detector}')
    logger.info(f'Matcher   : {config.components.matcher}')
    logger.info(f'Device    : {get_best_device()}')

    pipeline = DataPipeline()
    pipeline.run()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        console = Console()
        console.print_exception(show_locals=True)
