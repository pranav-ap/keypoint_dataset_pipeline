import random

import h5py
import numpy as np
from tqdm import tqdm

from config import config
from utils import logger


class TrainingDatasetCreator:
    def __init__(self):
        filename = 'train_data.hdf5'
        filepath = f'{config.paths[config.task.name].train_data}/{filename}'
        self._file = h5py.File(filepath, mode='w')

    @staticmethod
    def get_coords_on_original_image(reference_crop_coords, target_crop_coords):
        original_w, original_h = config.image.original_image_shape
        crop_w, crop_h = config.image.crop_image_shape

        left_padding = (original_w - crop_w) / 2
        top_padding = (original_h - crop_h) / 2

        reference_crop_coords = np.array([
            (kp[0] + left_padding, kp[1] + top_padding)
            for kp in reference_crop_coords
        ])

        target_crop_coords = np.array([
            (kp[0] + left_padding, kp[1] + top_padding)
            for kp in target_crop_coords
        ])

        return reference_crop_coords, target_crop_coords

    def extract_coords(self, refs_from, tars_from, refs_to, tars_to, indices_to):
        selected_pair_names = []

        for (pair_name, ref_dataset), (_, tar_dataset) in zip(refs_from.items(), tars_from.items()):
            if not isinstance(ref_dataset, h5py.Dataset) or not isinstance(tar_dataset, h5py.Dataset):
                continue

            reference_crop_coords = ref_dataset[()]
            target_crop_coords = tar_dataset[()]
            # assert len(reference_crop_coords) > 0, f'{pair_name} is empty'
            assert len(reference_crop_coords) == len(target_crop_coords)

            if len(reference_crop_coords) == 0:
                continue
            
            selected_pair_names.append(pair_name)

            reference_orig_coords, target_orig_coords = self.get_coords_on_original_image(
                reference_crop_coords,
                target_crop_coords
            )

            reference_coords_len = len(reference_orig_coords)
            target_coords_len = len(target_orig_coords)
            assert reference_coords_len > 0
            assert reference_coords_len == target_coords_len

            refs_to.create_dataset(pair_name, data=reference_orig_coords, compression='lzf')
            tars_to.create_dataset(pair_name, data=target_orig_coords, compression='lzf')

            patch_indices = random.sample(range(reference_coords_len), reference_coords_len)
            indices_to.create_dataset(pair_name, data=patch_indices, compression='lzf')

        return selected_pair_names

    @staticmethod
    def extract_rotations(rotations_from, rotations_to, selected_pair_names):
        for pair_name, rotations_dataset in rotations_from.items():
            if not isinstance(rotations_dataset, h5py.Dataset):
                continue

            if pair_name not in selected_pair_names:
                print(f'{pair_name} not found')
                continue

            rotations = rotations_dataset[()]
            # assert len(rotations) > 0, f'{pair_name} is empty'
            if len(rotations) == 0:
                continue

            rotations_to.create_dataset(pair_name, data=rotations, compression='lzf')

    def extract(self):
        for track in config.task.tracks:
            config.task.track = track
            logger.info(f'Track : {track}')
            config.task.dataset_kind = track[:2]

            filepath = f'{config.paths[config.task.name].output}/data.hdf5'
            # noinspection PyAttributeOutsideInit
            input_file = h5py.File(filepath, mode='r')

            for cam in tqdm(config.task.cams, total=2, desc="Extracting Original Coordinates", ncols=100):
                config.task.cam = cam
                logger.info(f'Cam : {cam}')

                refs_from = input_file[f'{cam}/matches/crop/reference_coords']
                tars_from = input_file[f'{cam}/matches/crop/target_coords']

                refs_to = self._file.create_group(f'{track}/{cam}/reference_coords')
                tars_to = self._file.create_group(f'{track}/{cam}/target_coords')
                indices_to = self._file.create_group(f'{track}/{cam}/indices')

                selected_pair_names = self.extract_coords(refs_from, tars_from, refs_to, tars_to, indices_to)

                rotations_from = input_file[f'{cam}/rotations']
                rotations_to = self._file.create_group(f'{track}/{cam}/rotations')

                self.extract_rotations(rotations_from, rotations_to, selected_pair_names)
        
            input_file.close()

        def print_hdf5_structure(f):
            def print_group(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"Group: {name}")

            f.visititems(print_group)
        
        print_hdf5_structure(self._file)
    
    def close(self):
        self._file.close()
