from config import config
from utils import get_best_device, make_clear_directory
from detectors import DeDoDeDetector
from matchers import RoMaMatcher
from typing import List, Tuple
import pandas as pd


class DataPipeline:
    def __init__(self):
        config.device = get_best_device()

        import sys
        sys.path.append('D:/thesis_code/keypoint_dataset_pipeline/RoMa')
        sys.path.append('D:/thesis_code/keypoint_dataset_pipeline/DeDoDe')

    @staticmethod
    def config_dedode_dedode_euroc():
        config.images_dir_path = "/kaggle/input/euroc-dataset/V1_01_easy/data"
        config.csv_path = "/kaggle/input/euroc-dataset/V1_01_easy/data.csv"
        config.filter_csv_path = ""

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
        config.filter_csv_path = ""

        config.npy_dir_path = "/kaggle/working/euroc-dataset/npy_files"
        make_clear_directory(config.npy_dir_path)

        config.POSTFIX_DATASET = 'euroc'
        config.POSTFIX_DETECTOR_MODEL = 'dedode'
        config.POSTFIX_MATCHER_MODEL = 'roma'

        config.IMAGE_RESIZE = (784, 784)

    @staticmethod
    def config_dedode_roma_matching_samples():
        config.images_dir_path = "/kaggle/input/matching-samples"
        config.csv_path = "/kaggle/input/matching-samples/data.csv"
        config.filter_csv_path = ""

        config.npy_dir_path = "/kaggle/working/matching-samples/npy_files"
        make_clear_directory(config.npy_dir_path)

        config.POSTFIX_DATASET = 'matching_samples'
        config.POSTFIX_DETECTOR_MODEL = 'dedode'
        config.POSTFIX_MATCHER_MODEL = 'roma'

        config.IMAGE_RESIZE = (784, 784)

    @staticmethod
    def config_dedode_roma_euroc_basalt():
        config.images_dir_path = "/kaggle/input/euroc-dataset/V1_01_easy/data"
        config.csv_path = "/kaggle/input/euroc-dataset/V1_01_easy/data.csv"
        config.filter_csv_path = "/kaggle/input/euroc-dataset/V1_01_easy/V1_01_easy.keyframes.csv"

        config.npy_dir_path = "/kaggle/working/euroc-dataset/npy_files"
        make_clear_directory(config.npy_dir_path)

        config.POSTFIX_DATASET = 'euroc'
        config.POSTFIX_DETECTOR_MODEL = 'dedode'
        config.POSTFIX_MATCHER_MODEL = 'roma'

        config.IMAGE_RESIZE = (784, 784)

    @staticmethod
    def get_sorted_image_names_list():
        df = pd.read_csv(config.csv_path)
        df['timestamp'] = pd.to_datetime(df['#timestamp [ns]'], unit='ns')
        df = df.sort_values(by='timestamp')
        image_names = df['filename'].tolist()

        return image_names

    @staticmethod
    def get_sorted_basalt_image_names_list():
        df = pd.read_csv(config.csv_path)
        df['timestamp'] = pd.to_datetime(df['#timestamp [ns]'], unit='ns')
        df = df.sort_values(by='timestamp')

        df_basalt = pd.read_csv(config.filter_csv_path, names=['timestamp'])
        df_basalt['timestamp'] = pd.to_datetime(df_basalt['timestamp'], unit='ns')

        df_filtered = df.query('timestamp in @df_basalt["timestamp"]')
        image_names = df_filtered['filename'].tolist()

        return image_names

    def get_sorted_image_name_pairs(self, step=1):
        image_names = self.get_sorted_image_names_list()
        image_pairs: List[Tuple[str, str]] = []

        for index in range(0, len(image_names) - step):
            name_a = image_names[index]
            name_b = image_names[index + step]

            image_pairs.append((name_a, name_b))

        return image_pairs

    def run(self):
        self.get_sorted_basalt_image_names_list()

        image_names = self.get_sorted_image_names_list()
        image_names = image_names[:5]

        detector = DeDoDeDetector()
        detector.extract_keypoints(image_names)

        matcher = RoMaMatcher()
        matcher.extract_warp_certainty(image_names)


def main():
    pipeline = DataPipeline()
    pipeline.run()


if __name__ == '__main__':
    main()
