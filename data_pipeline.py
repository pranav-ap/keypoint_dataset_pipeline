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

        if config.task.name != 'samples' or not config.task.consider_samples:
            make_clear_directory(config.paths[config.task.name].tensors_dir)

    @staticmethod
    def get_image_names():
        if config.task.name == 'samples' or config.task.consider_samples:
            return [
                str(config.samples[config.task.name].reference),
                str(config.samples[config.task.name].target)
            ]

        # only keypoints

        # df = pd.read_csv(config.paths[config.task.name].csv, header=None)
        # image_names = df[0].astype(str)

        # all images

        df = pd.read_csv(config.paths[config.task.name].csv)
        df['timestamp'] = pd.to_datetime(df['#timestamp [ns]'], unit='ns')
        df = df.sort_values(by='timestamp')
        df['filename'] = df['filename'].str.replace(".png", "", regex=False)
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

        project_root = '/home/stud/ath/ath_ws/keypoint_dataset_pipeline/'

        os.chdir(project_root)
        image_names = self.get_image_names()

        os.chdir(f'{project_root}/libs/DeDoDe')
        detector = DeDoDeDetector()
        detector.extract_keypoints(image_names)
        del detector

        os.chdir(f'{project_root}/libs/RoMa')
        matcher = RoMaMatcher()
        matcher.extract_warp_certainty(image_names)
        del matcher

        os.chdir(project_root)
        data_filter = DataFilter()
        data_filter.extract_good_matches(image_names)
        del data_filter

        if config.task.name != 'samples' or not config.task.consider_samples:
            folder_to_zip = config.paths[config.task.name].tensors_dir
            zipdir = config.paths[config.task.name].zip_dir
            output_zip_file = f'{zipdir}/matches.zip'
            wild = '_matches.pt'

            make_clear_directory(config.paths[config.task.name].zip_dir)
            zip_folder(folder_to_zip, output_zip_file, wild)

            config_filename = f'{zipdir}/current_config.yaml'
            OmegaConf.save(config, config_filename)
