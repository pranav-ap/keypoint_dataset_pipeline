from config import config
from src import TrainingDatasetCreator
from utils import logger, make_clear_directory


def main():
    logger.info(f'Keypoint Data Pipeline')

    logger.info(f'Clear Directory : {config.paths[config.task.name].train_data}')
    make_clear_directory(config.paths[config.task.name].train_data)

    creator = TrainingDatasetCreator()
    
    if config.task.only_missing:
        creator.extract_only_missing()
    else:
        creator.extract()
    
    creator.close()


if __name__ == '__main__':
    main()
