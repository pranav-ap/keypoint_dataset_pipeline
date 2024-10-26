from config import config
from utils import get_best_device, zip_folder, logger, make_clear_directory
from detectors import DeDoDeDetector
from matchers import RoMaMatcher
from DataFilter import DataFilter
from omegaconf import OmegaConf
import pandas as pd
import os


class DataPipeline:
    def __init__(self):
        self.device = get_best_device()
        make_clear_directory(config.paths[config.task.name].tensors_dir)

    @staticmethod
    def get_image_names():
        if config.task.consider_samples:
            return [
                config.samples[config.task.name].reference,
                config.samples[config.task.name].target
            ]

        df = pd.read_csv(config.paths[config.task.name].csv)
        image_names = df['filename'].tolist()

        if config.task.limit_count != 0:
            count = config.task.limit_count
            image_names = image_names[:count]

        return image_names

    def run(self):
        logger.info(f'Keypoint Data Pipeline')
        logger.info(f'Task      : {config.task.name}')
        logger.info(f'Detector  : {config.components.detector}')
        logger.info(f'Matcher   : {config.components.matcher}')
        logger.info(f'Device    : {get_best_device()}')

        image_names = self.get_image_names()

        os.chdir('/home/stud/ath/ath_ws/keypoint_dataset_pipeline/libs/DeDoDe')
        detector = DeDoDeDetector()
        detector.extract_keypoints(image_names)
        del detector

        os.chdir('/home/stud/ath/ath_ws/keypoint_dataset_pipeline/libs/RoMa')
        matcher = RoMaMatcher()
        matcher.extract_warp_certainty(image_names)
        del matcher

        os.chdir('/home/stud/ath/ath_ws/keypoint_dataset_pipeline')
        data_filter = DataFilter()
        data_filter.extract_good_matches(image_names)

        if not config.task.consider_samples:
            folder_to_zip = config.paths[config.task.name].tensors_dir
            zipdir = config.paths[config.task.name].zip_dir
            output_zip_file = f'{zipdir}/matches.zip'
            wild = '_matches.pt'

            make_clear_directory(config.paths[config.task.name].zip_dir)
            zip_folder(folder_to_zip, output_zip_file, wild)

            config_filename = f'{zipdir}/current_config.yaml'
            OmegaConf.save(config, config_filename)
