import random

import h5py
import numpy as np
from tqdm import tqdm

from config import config


class TrainingDatasetCreator:
    def __init__(self):
        filename = 'train_data.hdf5'
        filepath = f'{config.paths[config.task.name].train_data}/{filename}'
        self._file = h5py.File(filepath, mode='w')

    @staticmethod
    def get_coords_on_original_image(reference_crop_coords, target_crop_coords):
        original_w, original_h = config.image.original_image_shape
        crop_w, crop_h = config.image.crop_image_shape

        left_padding = (original_w - crop_w) // 2
        top_padding = (original_h - crop_h) // 2

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
        for (pair_name, ref_dataset), (_, tar_dataset) in zip(refs_from.items(), tars_from.items()):
            if not isinstance(ref_dataset, h5py.Dataset) or not isinstance(tar_dataset, h5py.Dataset):
                continue

            reference_crop_coords = ref_dataset[()]
            target_crop_coords = tar_dataset[()]

            reference_orig_coords, target_orig_coords = self.get_coords_on_original_image(
                reference_crop_coords,
                target_crop_coords
            )

            reference_coords_len = len(reference_orig_coords)
            target_coords_len = len(target_orig_coords)
            assert reference_coords_len == target_coords_len

            refs_to.create_dataset(pair_name, data=reference_orig_coords, compression='lzf')
            tars_to.create_dataset(pair_name, data=target_orig_coords, compression='lzf')

            N = min(reference_coords_len, config.train.num_patches_per_image)
            patch_indices = random.sample(range(reference_coords_len), N)

            indices_to.create_dataset(pair_name, data=patch_indices, compression='lzf')

    def extract(self):
        total = len(config.task.tracks)

        for track in tqdm(config.task.tracks, total=total, desc="Extracting Original Coordinates", ncols=100):
            config.task.track = track

            filepath = f'{config.paths[config.task.name].output}/data.hdf5'
            # noinspection PyAttributeOutsideInit
            input_file = h5py.File(filepath, mode='r')

            for cam in config.task.cams:
                config.task.cam = cam

                refs_from = input_file[f'{cam}/matches/crop/reference_coords']
                tars_from = input_file[f'{cam}/matches/crop/target_coords']

                refs_to = self._file.create_group(f'{track}/{cam}/reference_coords')
                tars_to = self._file.create_group(f'{track}/{cam}/target_coords')
                indices_to = self._file.create_group(f'{track}/{cam}/indices')

                self.extract_coords(refs_from, tars_from, refs_to, tars_to, indices_to)

            input_file.close()

    def close(self):
        self._file.close()
