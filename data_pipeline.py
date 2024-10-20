from config import config
from detectors import DeDoDeDetector
from matchers import RoMaMatcher
from DataFilter import DataFilter
import pandas as pd
import os


class DataPipeline:
    def __init__(self):
        from utils import make_clear_directory
        make_clear_directory(config.paths[config.task].tensors_dir)

    @staticmethod
    def get_image_names(count=None):
        if config.task == 'samples':
            return [config.samples.reference, config.samples.target]

        df = pd.read_csv(config.paths[config.task].csv_path)
        image_names = df['filename'].tolist()

        if count is not None:
            image_names = image_names[:count]

        return image_names

    def run(self):
        image_names = self.get_image_names()

        os.chdir('/home/stud/ath/ath_ws/keypoint_dataset_pipeline/DeDoDe')
        detector = DeDoDeDetector()
        detector.extract_keypoints(image_names)

        os.chdir('/home/stud/ath/ath_ws/keypoint_dataset_pipeline/RoMa')
        matcher = RoMaMatcher()
        matcher.extract_warp_certainty(image_names)

        os.chdir('/home/stud/ath/ath_ws/keypoint_dataset_pipeline/')
        data_filter = DataFilter()
        data_filter.extract_good_matches(image_names)
