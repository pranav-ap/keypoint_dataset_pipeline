import os

import pandas as pd
from omegaconf import OmegaConf

from config import config
from utils import get_best_device, logger, make_clear_directory
from .DataFilter import DataFilter
from .DataStore import DataStore
from .detectors import DeDoDeDetector
from .matchers import RoMaMatcher


class DataPipeline:
    def __init__(self):
        logger.info(f'Keypoint Data Pipeline')
        logger.info(f'Task      : {config.task.name}')
        logger.info(f'Detector  : {config.components.detector}')
        logger.info(f'Matcher   : {config.components.matcher}')
        logger.info(f'Device    : {get_best_device()}')

        if config.task.consider_samples:
            config.paths[config.task.name].output = config.paths.samples.output

        self.data_store = DataStore()

        os.chdir(f'{config.paths.roots.project}/libs/DeDoDe')
        self.detector = DeDoDeDetector(self.data_store)

        os.chdir(f'{config.paths.roots.project}/libs/RoMa')
        self.matcher = RoMaMatcher(self.data_store)

        os.chdir(config.paths.roots.project)
        self.data_filter = DataFilter(self.data_store)

    @staticmethod
    def get_image_names():
        if config.task.name == 'samples' or config.task.consider_samples:
            return [
                str(config.samples[config.task.name].reference),
                str(config.samples[config.task.name].target)
            ]

        df = pd.read_csv(config.paths[config.task.name].csv)
        df['timestamp'] = pd.to_datetime(df['#timestamp [ns]'], unit='ns')
        df = df.sort_values(by='timestamp')
        df['filename'] = df['filename'].str.replace(".png", "", regex=False)
        image_names = df['filename'].tolist()

        if config.task.limit_count != 0:
            count = config.task.limit_count
            image_names = image_names[:count]

        return image_names

    def _run(self):
        self.data_store.init()

        os.chdir(config.paths.roots.project)
        image_names = self.get_image_names()

        os.chdir(f'{config.paths.roots.project}/libs/DeDoDe')
        self.detector.extract_keypoints(image_names)

        os.chdir(f'{config.paths.roots.project}/libs/RoMa')
        self.matcher.extract_warp_certainty(image_names)

        os.chdir(config.paths.roots.project)
        self.data_filter.extract_good_matches(image_names)

    def run(self):
        logger.info('Data Pipeline has started running!')

        try:
            make_clear_directory(config.paths[config.task.name].output)
            config_filename = f'{config.paths[config.task.name].output}/config.yaml'
            OmegaConf.save(config, config_filename)

            if config.task.consider_samples:
                self._run()
            else:
                for cam in ['cam0', 'cam1']:
                    config.task.cam = cam
                    logger.info(f'Camera {cam}')
                    self._run()

        finally:
            self.data_store.close()

        logger.info('Data Pipeline has finished running!')
