import gc
import os
import shutil

import pandas as pd
import torch
from omegaconf import OmegaConf
from pathlib import Path
import random

from config import config
from utils import get_best_device, logger, make_clear_directory
from .DataFilter import DataFilter
from .DataStore import DataStore
from .detectors import DeDoDeDetector
from .matchers import RoMaMatcher


def pick_random_consecutive_pairs(lst, N):
    if len(lst) < 2 or N < 1:
        return []
    
    indices = random.sample(range(len(lst) - 1), N)
    return [(lst[i], lst[i + 1]) for i in indices]



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
        if config.task.consider_samples:
            return [
                str(config.samples[config.task.name].reference),
                str(config.samples[config.task.name].target)
            ]
        
        df = None

        if config.task.frame_filtering.startswith('all'):
            # use all frames
            df = pd.read_csv(config.paths[config.task.name].images_csv, header=0, names=('timestamp', 'filename'))            
        else:
            df = pd.read_csv(config.paths[config.task.name].keyframes_csv, header=0, names=('timestamp', 'filename', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz'), usecols=('timestamp', 'filename'))
        
        if len(df) == 0:
            return []
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
        df = df.sort_values(by='timestamp')

        def image_exists(filename):
            image_path = Path(f"{config.paths[config.task.name].images}/{filename.strip()}")
            return image_path.exists()

        df = df[df['filename'].apply(image_exists)].reset_index(drop=True)

        df['filename'] = df['filename'].str.replace(".png", "", regex=False)
        image_names = df['filename'].tolist()
        logger.debug(f'Before {len(image_names)=}')

        if config.task.only_missing:
            N = len(image_names)
            
            if N < 500:
                N = 150
            elif N < 1000:
                N = 300
            elif N < 3000:
                N = 400
            else:
                N = 600
            
            N = min(N, len(image_names))
            
            if config.task.limit_count != 0:
                N = min(N, config.task.limit_count)

            if N == len(image_names):
                N -= 1
                
            random_pairs = pick_random_consecutive_pairs(image_names, N)
            logger.info(f'{len(random_pairs)=}')
            return random_pairs

        if config.task.limit_count != 0:
            count = config.task.limit_count
            image_names = image_names[:count]
            logger.info(f'{len(image_names)=}')
        
        return image_names

    def _process_images(self):
        self.data_store.init()

        image_names = self.get_image_names()

        # if len(image_names) == 0:
        #     return

        self.detector.extract_keypoints(image_names)
        self.matcher.extract_warp_certainty(image_names)
        self.data_filter.extract_good_matches(image_names)

    def _process_images_missing(self):
        self.data_store.init(for_all_missing=True)
        
        random_pairs = self.get_image_names()

        self.matcher.extract_warp_certainty_missing(random_pairs)
        self.data_store.close()

    def run(self):
        logger.info('Data Pipeline has started running!')
        torch.cuda.empty_cache()

        try:
            config.task.dataset_kind = config.task.track[:2]

            if config.task.frame_filtering.startswith('all'):
                logger.info(f'Clear Directory : {config.paths[config.task.name].output}')
                make_clear_directory(config.paths[config.task.name].output)

            logger.info('Save Current Config')
            config_filepath = f'{config.paths[config.task.name].output}/config.yaml'
            OmegaConf.save(config, config_filepath)

            if config.task.consider_samples:
                self._process_images()
            else:
                # Copy IMU Data
                shutil.copy(
                    config.paths[config.task.name].imu_csv,
                    f'{config.paths[config.task.name].output}/imu_data.csv'
                )

                # Process Images
                for cam in config.task.cams:
                    if cam == 'cam2' or cam == 'cam3':
                        if not track.startswith('MG'):
                            continue

                    logger.info(f'Camera {cam}')
                    config.task.cam = cam
                    # torch.cuda.empty_cache()
                    gc.collect()
                    self._process_images()

        finally:
            self.data_store.close()

        logger.info('Data Pipeline has finished running!')

    def run_list(self):
        logger.info('Data Pipeline has started running!')
        torch.cuda.empty_cache()

        for track in config.task.tracks:
            config.task.track = track

            try:
                logger.info(f'Filter {track}')
                config.task.dataset_kind = track[:2]

                if config.task.frame_filtering.startswith('all'):
                    logger.info(f'Clear Directory : {config.paths[config.task.name].output}')
                    make_clear_directory(config.paths[config.task.name].output)

                logger.info('Save Current Config')
                config_filepath = f'{config.paths[config.task.name].output}/config.yaml'
                OmegaConf.save(config, config_filepath)

                if config.task.consider_samples:
                    self._process_images()
                else:
                    # Copy IMU Data
                    shutil.copy(
                        config.paths[config.task.name].imu_csv,
                        f'{config.paths[config.task.name].output}/imu_data.csv'
                    )

                    # Process Images
                    for cam in config.task.cams:
                        if cam == 'cam2' or cam == 'cam3':
                            if not track.startswith('MG'):
                                continue

                        logger.info(f'Camera {cam}')
                        config.task.cam = cam
                        # torch.cuda.empty_cache()
                        gc.collect()
                        self._process_images()

            finally:
                self.data_store.close()

        logger.info('Data Pipeline has finished running!')

    def run_missing_list(self):
        logger.info('Data Pipeline has started running!')
        torch.cuda.empty_cache()

        for track in config.task.tracks:
            config.task.track = track

            try:
                logger.info(f'Running on {track}')
                config.task.dataset_kind = config.task.track[:2]

                logger.info(f'Clear Directory : {config.paths[config.task.name].output}')
                make_clear_directory(config.paths[config.task.name].output)

                logger.info('Save Current Config')
                config_filepath = f'{config.paths[config.task.name].output}/config.yaml'
                OmegaConf.save(config, config_filepath)

                # Copy IMU Data
                shutil.copy(
                    config.paths[config.task.name].imu_csv,
                    f'{config.paths[config.task.name].output}/imu_data.csv'
                )

                # Process Images
                for cam in config.task.cams:
                    if cam == 'cam2' or cam == 'cam3':
                        if not track.startswith('MG'):
                            continue

                    logger.info(f'Camera {cam}')
                    config.task.cam = cam
                    # torch.cuda.empty_cache()
                    gc.collect()
                    self._process_images_missing()

            finally:
                self.data_store.close()

        logger.info('Data Pipeline has finished running!')
