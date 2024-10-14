from utils.logger import logger
from config import config
import pandas as pd


class DataPipeline:
    def __init__(self):
        pass

    @staticmethod
    def _get_sorted_image_names():
        df = pd.read_csv(config.csv_path)
        df['timestamp'] = pd.to_datetime(df['#timestamp [ns]'], unit='ns')
        df = df.sort_values(by='timestamp')
        image_names = df['filename'].tolist()

        return image_names

    def run(self):
        config.images_dir_path = "/kaggle/input/euroc-v1-01-easy/V1_01_easy/data"
        config.csv_path = "/kaggle/input/euroc-v1-01-easy/V1_01_easy/data.csv"
        config.npy_dir_path = "/kaggle/working/euroc-v1-01-easy/npy_files"

        image_names = self._get_sorted_image_names()
        # image_names = image_names[:5]

        self.detect_describe_match(image_names)
        self.load_and_visualize(image_names)


def main():
    pipeline = DataPipeline()
    pipeline.run()


if __name__ == '__main__':
    main()
