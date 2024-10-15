from utils.logger import logger
from config import config
from utils import get_best_device
from detectors import DeDoDeDetector
from matchers import DeDoDeMatcher
import pandas as pd


class DataPipeline:
    def __init__(self):
        config.device = get_best_device()
        config.POSTFIX_FILE = f'{config.POSTFIX_MODEL}_{config.POSTFIX_DATASET}'

        config.images_dir_path = "/kaggle/input/euroc-v1-01-easy/V1_01_easy/data"
        config.csv_path = "/kaggle/input/euroc-v1-01-easy/V1_01_easy/data.csv"
        config.npy_dir_path = "/kaggle/working/euroc-v1-01-easy/npy_files"

    @staticmethod
    def _get_sorted_image_names():
        df = pd.read_csv(config.csv_path)
        df['timestamp'] = pd.to_datetime(df['#timestamp [ns]'], unit='ns')
        df = df.sort_values(by='timestamp')
        image_names = df['filename'].tolist()

        return image_names

    def roma_match(self):
        config.IMAGE_RESIZE = (self.H, self.W)

    def run(self):
        image_names = self._get_sorted_image_names()
        image_names = image_names[:5]

        detector = DeDoDeDetector(image_names)
        detector.extract_keypoints()

        matcher = DeDoDeMatcher(image_names)
        matcher.extract_matches()


def main():
    pipeline = DataPipeline()
    pipeline.run()


if __name__ == '__main__':
    main()
