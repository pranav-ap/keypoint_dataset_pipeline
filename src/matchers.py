from abc import ABC
from typing import Optional
import numpy as np
import torch
from skimage import feature
import pandas as pd
import random
from collections import defaultdict 

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


def crop_image_alb(image: Image.Image, keypoint, left, upper, right, lower, patch_size=32):
    patch = image.crop((left, upper, right, lower))
    assert patch.size[0] == patch.size[1]
    assert patch.size[0] == patch_size

    new_keypoint = keypoint[0] - left, keypoint[1] - upper

    return patch, new_keypoint


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


def moran_skip(crop1, threshold=0.7):
    crop1_np = np.array(crop1)

    # Convert image to 1D array
    flattened_patch = crop1_np.ravel()

    # Create a spatial weight matrix
    w = lat2W(*crop1_np.shape)
    moran = Moran(flattened_patch, w)

    must_skip = abs(moran.I) < threshold  

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

        self.res = (config.image.crop_image_shape[1], config.image.crop_image_shape[0])

        if config.task.only_missing:
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
        min_edge_density_threshold = 0.04
        seen_kpids = defaultdict(int)
        desired_patch_size = config.image.patch_size
        perturb_size = 2

        for name_a, name_b in tqdm(
            random_pairs,
            desc="Extracting warps", 
            ncols=100,
            total=len(random_pairs),
            ):

            a = Keypoints(name_a, self.data_store)
            b = Keypoints(name_b, self.data_store)

            df = pd.read_csv(
                f"/home/stud/ath/ath_ws/datasets/track_debug_3_lifetimes/{config.task.track}/{config.task.cam}/{name_a}_incoming_missed_kps.csv",
                header=0, names=("kpid", "x", "y", "x_guess", "y_guess")
            )
           
            # N = config.roma.filter.missed_kp_count
            # N = min(N, len(df))
            # shuffled_df = df.sample(n=N)

            for index, row in df.iterrows():
                kpid = row['kpid']
            
                # if seen_kpids[kpid] >= 30:
                #     continue

                x, y, x_guess, y_guess = row['x'], row['y'], row['x_guess'], row['y_guess']

                # perturb_x = np.random.randint(1, perturb_size) * np.random.choice([1, -1])
                # perturb_y = np.random.randint(1, perturb_size) * np.random.choice([1, -1])

                ref_center = [x, y]
                
                # if 0 <= x + perturb_x < config.image.original_image_shape[0]:
                #     ref_center[0] = x + perturb_x

                # if 0 <= y + perturb_y < config.image.original_image_shape[1]:
                #     ref_center[1] = y + perturb_y 

                # left, upper, right, lower are always within given image dimensions
                ref_left, ref_upper, ref_right, ref_lower = get_patch_boundary(
                    a.original_image, 
                    ref_center, 
                    patch_size=desired_patch_size
                )

                ref_keypoint = [x, y]
                ref_crop, ref_crop_keypoint = crop_image_alb(
                    a.original_image, 
                    ref_keypoint, 
                    ref_left, ref_upper, ref_right, ref_lower, 
                    patch_size=desired_patch_size,
                )

                must_skip = edge_skip(
                    ref_crop,
                    min_edge_density_threshold=min_edge_density_threshold,
                )

                if must_skip:
                    continue

                seen_kpids[kpid] += 1

                perturb_x = np.random.randint(1, perturb_size) * np.random.choice([1, -1])
                perturb_y = np.random.randint(1, perturb_size) * np.random.choice([1, -1])

                tar_center = [x_guess, y_guess]

                if 2 <= x_guess + perturb_x < config.image.original_image_shape[0] - 2:
                    tar_center[0] = x_guess + perturb_x 

                if 2 <= y_guess + perturb_y < config.image.original_image_shape[1] - 2:
                    tar_center[1] = y_guess + perturb_y
                
                tar_left, tar_upper, tar_right, tar_lower = get_patch_boundary(
                    b.original_image, 
                    tar_center, 
                    patch_size=desired_patch_size
                )

                tar_keypoint = [x_guess, y_guess]
                tar_crop, tar_crop_keypoint = crop_image_alb(
                    b.original_image, 
                    tar_keypoint, 
                    tar_left, tar_upper, tar_right, tar_lower, 
                    patch_size=desired_patch_size,
                )

                # Match using model and retrieve warp and certainty
                warp, certainty = self.model.match(
                    ref_crop.convert('RGB'), tar_crop.convert('RGB'),
                    device=self.device
                )

                saves = [
                    ref_crop_keypoint[0], ref_crop_keypoint[1],
                    ref_left, ref_upper, ref_right, ref_lower,

                    tar_crop_keypoint[0], tar_crop_keypoint[1],
                    tar_left, tar_upper, tar_right, tar_lower,
                ]

                # Set warp and certainty for the match pair
                pair = Matches(a, b, self.data_store, kpid=kpid, saves=saves)
                pair.set_warp(warp)
                pair.certainty = certainty
                pair.save()

