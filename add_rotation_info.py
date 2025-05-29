from config import config
from src import RotationInfoWriter
from utils import logger


def main():
    logger.info(f'Add Rotation Info')

    writer = RotationInfoWriter()
    writer.extract()


if __name__ == '__main__':
    main()
