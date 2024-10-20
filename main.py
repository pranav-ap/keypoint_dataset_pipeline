from utils import logger
from config import config
from data_pipeline import DataPipeline


def main():
    logger.info(f'Keypoint Data Pipeline')
    logger.info(f'Task      : {config.task}')
    logger.info(f'Detector  : {config.components.detector}')
    logger.info(f'Matcher   : {config.components.matcher}')

    pipeline = DataPipeline()
    pipeline.run()


if __name__ == '__main__':
    main()
