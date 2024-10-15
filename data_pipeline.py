from config import config
from utils import get_best_device, make_clear_directory
from detectors import DeDoDeDetector
from matchers import DeDoDeMatcher, RoMaMatcher
import pandas as pd


class DataPipeline:
    def __init__(self):
        config.device = get_best_device()
        config.images_dir_path = "/kaggle/input/euroc-v1-01-easy/V1_01_easy/data"
        config.csv_path = "/kaggle/input/euroc-v1-01-easy/V1_01_easy/data.csv"

        config.npy_dir_path = "/kaggle/working/euroc-v1-01-easy/npy_files"
        make_clear_directory(config.npy_dir_path)

        config.POSTFIX_DATASET = 'euroc'

    @staticmethod
    def get_sorted_image_names():
        df = pd.read_csv(config.csv_path)
        df['timestamp'] = pd.to_datetime(df['#timestamp [ns]'], unit='ns')
        df = df.sort_values(by='timestamp')
        image_names = df['filename'].tolist()

        return image_names

    def run(self):
        image_names = self.get_sorted_image_names()
        image_names = image_names[:5]

        config.POSTFIX_MODEL = 'dedode'
        detector = DeDoDeDetector()
        detector.extract_keypoints(image_names)

        # config.POSTFIX_MODEL = 'roma'
        # matcher = RoMaMatcher()
        # config.IMAGE_RESIZE = (matcher.W, matcher.H)
        # matcher.extract_matches(image_names)

        config.POSTFIX_MODEL = 'dedode'
        matcher = DeDoDeMatcher()
        config.IMAGE_RESIZE = (784, 784)
        matcher.extract_matches(image_names)


def main():
    pipeline = DataPipeline()
    pipeline.run()


if __name__ == '__main__':
    main()
