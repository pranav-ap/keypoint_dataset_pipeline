from config import config
from utils import get_best_device, make_clear_directory
from detectors import DeDoDeDetector
from matchers import DeDoDeMatcher, RoMaMatcher
import pandas as pd


class DataPipeline:
    def __init__(self):
        config.device = get_best_device()

    @staticmethod
    def config_dedode_dedode_euroc():
        config.images_dir_path = "/kaggle/input/euroc-dataset/V1_01_easy/data"
        config.csv_path = "/kaggle/input/euroc-dataset/V1_01_easy/data.csv"

        config.npy_dir_path = "/kaggle/working/euroc-dataset/npy_files"
        make_clear_directory(config.npy_dir_path)

        config.POSTFIX_DATASET = 'euroc'
        config.POSTFIX_DETECTOR_MODEL = 'dedode'
        config.POSTFIX_MATCHER_MODEL = 'dedode'

        config.IMAGE_RESIZE = (784, 784)

    @staticmethod
    def config_dedode_roma_euroc():
        config.images_dir_path = "/kaggle/input/euroc-dataset/V1_01_easy/data"
        config.csv_path = "/kaggle/input/euroc-dataset/V1_01_easy/data.csv"

        config.npy_dir_path = "/kaggle/working/euroc-dataset/npy_files"
        make_clear_directory(config.npy_dir_path)

        config.POSTFIX_DATASET = 'euroc'
        config.POSTFIX_DETECTOR_MODEL = 'dedode'
        config.POSTFIX_MATCHER_MODEL = 'roma'

    @staticmethod
    def config_dedode_roma_matching_samples():
        config.images_dir_path = "/kaggle/input/matching-samples"
        config.csv_path = "/kaggle/input/matching-samples/data.csv"

        config.npy_dir_path = "/kaggle/working/matching-samples/npy_files"
        make_clear_directory(config.npy_dir_path)

        config.POSTFIX_DATASET = 'matching_samples'
        config.POSTFIX_DETECTOR_MODEL = 'dedode'
        config.POSTFIX_MATCHER_MODEL = 'roma'

    @staticmethod
    def get_sorted_image_names():
        df = pd.read_csv(config.csv_path)
        df['timestamp'] = pd.to_datetime(df['#timestamp [ns]'], unit='ns')
        df = df.sort_values(by='timestamp')
        image_names = df['filename'].tolist()

        return image_names

    def run(self):
        self.config_dedode_dedode_euroc()

        image_names = self.get_sorted_image_names()
        image_names = image_names[:5]

        detector = DeDoDeDetector()
        detector.extract_keypoints(image_names)

        # matcher = DeDoDeMatcher()
        # matcher.extract_matches(image_names)

        matcher = RoMaMatcher()
        matcher.extract_matches(image_names)


def main():
    pipeline = DataPipeline()
    pipeline.run()


if __name__ == '__main__':
    main()
