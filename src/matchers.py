from abc import ABC
from typing import Optional
import numpy as np
from skimage import feature
import pandas as pd
import random

from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw, ImageFont

from config import config
from utils import get_best_device, logger
from .ImageData import Keypoints, Matches


def get_patch_boundary(image: Image.Image, center_point, patch_size):
    image_width, image_height = image.size
    x, y = center_point
    half_patch_size = patch_size // 2

    left, right = x - half_patch_size, x + half_patch_size
    upper, lower = y - half_patch_size, y + half_patch_size

    if left < 0:
        right += -left
        left = 0
    elif right > image_width:
        left -= right - image_width
        right = image_width

    if upper < 0:
        lower += -upper
        upper = 0
    elif lower > image_height:
        upper -= lower - image_height
        lower = image_height

    assert right > left
    assert right - left == patch_size
    assert lower > upper
    assert lower - upper == patch_size

    return left, upper, right, lower


def crop_image_alb(image: Image.Image, keypoint, patch_size=32):
    left, upper, right, lower = get_patch_boundary(image, keypoint, patch_size)

    patch = image.crop((left, upper, right, lower))
    assert patch.size[0] == patch.size[1]
    assert patch.size[0] == patch_size

    new_keypoint = keypoint[0] - left, keypoint[1] - upper

    return patch, new_keypoint, left, upper



def edge_skip(crop1, min_edge_density_threshold=0.012):
    p = crop1.copy().convert("L")
    # p = p.filter(ImageFilter.MedianFilter(size=3))

    patch_array = np.array(p)

    sigma = 1
    edges = (
        feature.canny(
            patch_array,
            sigma=sigma,
        ).astype(np.uint8)
        * 255
    )

    # Count non-zero pixels (edges)
    num_edges = np.count_nonzero(edges)
    total_pixels = patch_array.shape[0] * patch_array.shape[1]
    edge_density = round(num_edges / total_pixels, 4)

    must_skip = edge_density < min_edge_density_threshold

    return must_skip


class KeypointMatcher(ABC):
    device = get_best_device()

    def __init__(self, data_store):
        self.data_store = data_store


class RoMaMatcher(KeypointMatcher):
    def __init__(self, data_store):
        super().__init__(data_store)

        logger.info('Loading RoMaMatcher')
        from romatch import roma_outdoor

        # self.res = (config.image.crop_image_shape[1], config.image.crop_image_shape[0])
        self.res = (config.image.patch_size, config.image.patch_size)

        self.model = roma_outdoor(
            device=self.device,
            # (height, width)
            coarse_res=420,
            upsample_res=self.res,
        )

        self.model.symmetric = False
        logger.info('Loading RoMaMatcher Done')

    def extract_warp_certainty(self, image_names):
        a: Optional[Keypoints] = None

        for name_a, name_b in tqdm(
            zip(image_names, image_names[1:]), 
            desc="Extracting warps", 
            ncols=100,
            total=len(image_names) - 1,
            ):
            if a is None:
                a = Keypoints(name_a, self.data_store)

            b = Keypoints(name_b, self.data_store)

            # Match using model and retrieve warp and certainty
            warp, certainty = self.model.match(
                a.image, b.image,
                device=self.device
            )

            # Set warp and certainty for the match pair
            pair = Matches(a, b, self.data_store)
            pair.set_warp(warp)
            pair.certainty = certainty
            pair.save()

            # Move forward
            a = b

    def extract_warp_certainty_missing(self, random_pairs):
        min_edge_density_threshold = 0.02

        for name_a, name_b in tqdm(
            random_pairs,
            desc="Extracting warps", 
            ncols=100,
            total=len(random_pairs) - 1,
            ):
            
            a = Keypoints(name_a, self.data_store)
            b = Keypoints(name_b, self.data_store)

            df = pd.read_csv(
                f"/home/stud/ath/ath_ws/datasets/track_debug/{config.task.track}/{config.task.cam}/{a.image_name}_incoming_missed_kps.csv",
                header=0, names=("kpid", "x", "y", "x_guess", "y_guess")
            )
           
            # N = config.roma.filter.missed_kp_count
            # N = min(N, len(df))
            # shuffled_df = df.sample(n=N)

            for index, row in df.iterrows():
                # print(f'kpid={row['kpid']}')

                x, y, x_guess, y_guess = row['x'], row['y'], row['x_guess'], row['y_guess']
                # print(f'{x, y, x_guess, y_guess=}')

                left_crop, left_crop_keypoint, left, upper = crop_image_alb(
                    a.original_image, [x, y], patch_size=config.image.patch_size,
                )

                must_skip = edge_skip(
                    left_crop,
                    min_edge_density_threshold=min_edge_density_threshold,
                )

                if must_skip and random.random() > 0.1:
                    continue

                right_crop, right_crop_keypoint, left, upper = crop_image_alb(
                    b.original_image, [x_guess, y_guess], patch_size=config.image.patch_size,
                )

                # logger.debug(f'{a.original_image.size=}')
                # logger.debug(f'{left_crop.size=}')

                assert a.original_image.size == b.original_image.size, 'images must have same size'
                assert left_crop.size == right_crop.size, 'crops must have same size'

                # Match using model and retrieve warp and certainty
                # print('run roma')
                warp, certainty = self.model.match(
                    left_crop.convert('RGB'), right_crop.convert('RGB'),
                    device=self.device
                )
                # print('roma done')

                # Set warp and certainty for the match pair
                pair = Matches(a, b, self.data_store, kpid=row['kpid'])
                pair.set_warp(warp)
                pair.certainty = certainty
                pair.save()

