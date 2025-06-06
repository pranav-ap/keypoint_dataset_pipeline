import random

import h5py
import numpy as np
from tqdm import tqdm
import numpy as np
import PIL.Image

from .rotate import solve_patch_rotation
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


class RotationInfoWriter:
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

    def extract_rot(self, refs_from, tars_from, rotations_to):
        for pair_name in refs_from.keys() & tars_from.keys():
            ref_dataset = refs_from[pair_name]
            tar_dataset = tars_from[pair_name]

            if not isinstance(ref_dataset, h5py.Dataset) or not isinstance(tar_dataset, h5py.Dataset):
                continue

            # print(f'{pair_name=}')

            reference_crop_coords = ref_dataset[()]
            target_crop_coords = tar_dataset[()]
            assert len(reference_crop_coords) == len(target_crop_coords)

            if len(reference_crop_coords) == 0:
                continue

            reference_orig_coords, target_orig_coords = self.get_coords_on_original_image(
                reference_crop_coords,
                target_crop_coords
            )

            reference_coords_len = len(reference_orig_coords)
            target_coords_len = len(target_orig_coords)
            assert reference_coords_len > 0
            assert reference_coords_len == target_coords_len

            # Rotation

            a, b = pair_name.split('_')

            image_path_a: str = f"{config.paths[config.task.name].images}/{a}.png"
            image_path_b: str = f"{config.paths[config.task.name].images}/{b}.png"
            
            rotations = []

            img0 = np.array(PIL.Image.open(image_path_a), dtype=np.uint8)
            img1 = np.array(PIL.Image.open(image_path_b), dtype=np.uint8)
            
            for i in range(len(reference_orig_coords)):
                angle = solve_patch_rotation(
                    img0, img1,
                    np.array([reference_orig_coords[i][0], reference_orig_coords[i][1]]),
                    np.array([target_orig_coords[i][0], target_orig_coords[i][1]]),
                )

                rotations.append(angle)

            assert len(reference_orig_coords) == len(rotations), f'{pair_name=} causes trouble'

            rotations_to.create_dataset(pair_name, data=rotations, compression='lzf')

    def extract(self):
        for track in config.task.tracks:
            config.task.track = track
            logger.info(f'Track : {track}')
            config.task.dataset_kind = track[:2]

            filepath = f'{config.paths[config.task.name].output}/data.hdf5'
            filepath = filepath.replace('_test', '')
            print(filepath)

            # noinspection PyAttributeOutsideInit
            f = h5py.File(filepath, mode='r+')

            print_hdf5_structure(f)

            for cam in tqdm(config.task.cams, total=2, desc="Extracting Rotation values", ncols=100):
                config.task.cam = cam
                logger.info(f'Cam : {cam}')

                refs_from = f[f'{cam}/matches/crop/reference_coords']
                tars_from = f[f'{cam}/matches/crop/target_coords']
                
                if f'{cam}/rotationssss' in f:
                    del f[f'{cam}/rotationssss']

                rotations_to = f.create_group(f'{cam}/rotationssss')

                self.extract_rot(refs_from, tars_from, rotations_to)
        
            f.close()
        