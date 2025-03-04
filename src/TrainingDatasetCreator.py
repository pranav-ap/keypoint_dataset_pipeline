import random

import h5py
import numpy as np
from tqdm import tqdm
import gc

from config import config
from utils import logger


def print_hdf5_structure(f):
    def print_group(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    try:
        f.visititems(print_group)
    except RuntimeError as e:
        print(f"Skipping corrupted object: {e}")

    # f.visititems(print_group)



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

    def extract_coords_and_rots(self, refs_from, tars_from, rotations_from, refs_to, tars_to, indices_to, rotations_to):
        for pair_name in refs_from.keys() & tars_from.keys() & rotations_from.keys():
            ref_dataset = refs_from[pair_name]
            tar_dataset = tars_from[pair_name]
            rot_dataset = rotations_from[pair_name]

            if not isinstance(ref_dataset, h5py.Dataset) or not isinstance(tar_dataset, h5py.Dataset) or not isinstance(rot_dataset, h5py.Dataset):
                continue

            # print(f'{pair_name=}')

            reference_crop_coords = ref_dataset[()]
            target_crop_coords = tar_dataset[()]
            rot_values = rot_dataset[()]

            if len(reference_crop_coords) == 0:
                continue

            assert len(reference_crop_coords) == len(target_crop_coords) == len(rot_values), f'Mismatched Shape - {pair_name=} : {reference_crop_coords.shape=} : {rot_values.shape=}'

            # if len(reference_crop_coords) == len(target_crop_coords) == len(rot_values):
            #     pass
            # else:
            #     print(f'Mismatched Shape - {pair_name=} : {reference_crop_coords.shape=} : {rot_values.shape=}')
               
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

            # random.sample(range(reference_coords_len), reference_coords_len)
            patch_indices = list(range(reference_coords_len)) 
            indices_to.create_dataset(pair_name, data=patch_indices, compression='lzf')

            rotations_to.create_dataset(pair_name, data=rot_values, compression='lzf')

    def extract_coords(self, refs_from, tars_from, refs_to, tars_to, indices_to):
        for pair_name in refs_from.keys() & tars_from.keys():
            ref_dataset = refs_from[pair_name]
            tar_dataset = tars_from[pair_name]

            if not isinstance(ref_dataset, h5py.Dataset) or not isinstance(tar_dataset, h5py.Dataset):
                continue

            # print(f'{pair_name=}')

            reference_crop_coords = ref_dataset[()]
            target_crop_coords = tar_dataset[()]

            if len(reference_crop_coords) == 0:
                continue

            assert len(reference_crop_coords) == len(target_crop_coords), f'Mismatched Shape - {pair_name=} : {reference_crop_coords.shape=} : {rot_values.shape=}'

            # if len(reference_crop_coords) == len(target_crop_coords) == len(rot_values):
            #     pass
            # else:
            #     print(f'Mismatched Shape - {pair_name=} : {reference_crop_coords.shape=} : {rot_values.shape=}')
               
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

            # random.sample(range(reference_coords_len), reference_coords_len)
            patch_indices = list(range(reference_coords_len)) 
            indices_to.create_dataset(pair_name, data=patch_indices, compression='lzf')

    def extract(self):
        for track in config.task.tracks:
            config.task.track = track
            logger.info(f'Track : {track}')
            config.task.dataset_kind = track[:2]

            filepath = f'{config.paths[config.task.name].output}/data.hdf5'
            filepath = filepath.replace('_test', '')
            # noinspection PyAttributeOutsideInit
            input_file = h5py.File(filepath, mode='r')

            # print_hdf5_structure(input_file)

            for cam in tqdm(config.task.cams, total=2, desc="Extracting Original Coordinates", ncols=100):
                if cam == 'cam2' or cam == 'cam3':
                    if not track.startswith('MG'):
                        continue

                config.task.cam = cam
                logger.info(f'Cam : {cam}')

                refs_from = input_file[f'{cam}/matches/crop/reference_coords']
                tars_from = input_file[f'{cam}/matches/crop/target_coords']
                # rotations_from = input_file[f'{cam}/rotationssss']  # ssss not s

                refs_to = self._file.create_group(f'{track}/{cam}/reference_coords')
                tars_to = self._file.create_group(f'{track}/{cam}/target_coords')
                indices_to = self._file.create_group(f'{track}/{cam}/indices')
                # rotations_to = self._file.create_group(f'{track}/{cam}/rotations') # s not ssss

                # self.extract_coords_and_rots(refs_from, tars_from, rotations_from, refs_to, tars_to, indices_to, rotations_to)
                self.extract_coords(refs_from, tars_from, refs_to, tars_to, indices_to)
        
            input_file.close()

        print_hdf5_structure(self._file)
        print('Done!')

    def extract_warps_only_missing(self, warp_from, cert_from, saves_from, warp_to, cert_to, saves_to):
        for pair_name in warp_from.keys() & cert_from.keys() & saves_from.keys():
            warp_dataset = warp_from[pair_name]
            cert_dataset = cert_from[pair_name]
            saves_dataset = saves_from[pair_name]

            if not isinstance(warp_dataset, h5py.Dataset) or not isinstance(cert_dataset, h5py.Dataset) or not isinstance(saves_dataset, h5py.Dataset):
                continue

            # print(f'{pair_name=}')

            warp = warp_dataset[()]
                
            # gc.collect()

            try:
                cert = cert_dataset[()]
            except Exception as e:
                logger.debug('oops')
                continue

            # gc.collect()

            saves = saves_dataset[()]

            warp_to.create_dataset(pair_name, data=warp, compression='lzf')
            cert_to.create_dataset(pair_name, data=cert, compression='lzf')
            saves_to.create_dataset(pair_name, data=saves, compression='lzf')

    def extract_only_missing(self):
        gc.collect()
        
        for track in config.task.tracks:
            config.task.track = track
            logger.info(f'Track : {track}')
            config.task.dataset_kind = track[:2]

            filepath = f'{config.paths[config.task.name].output}/data.hdf5'
            # filepath = filepath.replace('_test', '')
            # noinspection PyAttributeOutsideInit
            input_file = h5py.File(filepath, mode='r')

            # print_hdf5_structure(input_file)

            total = 2 if not track.startswith('MG') else 4 

            for cam in tqdm(config.task.cams, total=total, desc="Transferring to train_data file", ncols=100):
                if cam == 'cam2' or cam == 'cam3':
                    if not track.startswith('MG'):
                        continue
                
                # gc.collect()

                config.task.cam = cam
                logger.info(f'Cam : {cam}')

                warp_from = input_file[f'{cam}/matcher/warp']
                cert_from = input_file[f'{cam}/matcher/certainty']
                saves_from = input_file[f'{cam}/matcher/saves']

                warp_to = self._file.create_group(f'{track}/{cam}/matcher/warp')
                cert_to = self._file.create_group(f'{track}/{cam}/matcher/certainty')
                saves_to = self._file.create_group(f'{track}/{cam}/matcher/saves')

                # gc.collect()
                self.extract_warps_only_missing(warp_from, cert_from, saves_from, warp_to, cert_to, saves_to)
                # gc.collect()
        
        
            # gc.collect()
            input_file.close()
            # gc.collect()

        print_hdf5_structure(self._file)
        print('Done!')

    def close(self):
        self._file.close()
