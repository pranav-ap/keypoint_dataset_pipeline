from utils.logger import logger
from config import config
from utils import get_best_device
from ImageData import ImageSoloData, ImagePairData
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from typing import Optional
from abc import ABC, abstractmethod


class KeypointMatcher(ABC):
    def __init__(self, file_postfix):
        self.device = get_best_device()
        self.file_postfix = file_postfix

    @abstractmethod
    def match(self):
        pass

    @abstractmethod
    def match_pair(self, pair: ImagePairData):
        pass


class DeDoDeMatcher(KeypointMatcher):
    def __init__(self, image_names):
        super().__init__()
        self.image_names = image_names

        from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
        self.matcher = DualSoftMaxMatcher()

    """
    Visualization
    """

    @staticmethod
    def plot_image(image):
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Plot the image
        plt.figure(figsize=(30, 30))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

    @staticmethod
    def draw_matches(a, left_matches, b, right_matches):
        left_matches = [cv2.KeyPoint(x, y, 1.) for x, y in left_matches]
        right_matches = [cv2.KeyPoint(x, y, 1.) for x, y in right_matches]

        count = len(left_matches)
        matches = [cv2.DMatch(idx, idx, 0.) for idx in range(count)]

        a, b = np.array(a), np.array(b)

        match_image = cv2.drawMatches(
            a, left_matches,
            b, right_matches,
            matches,
            outImg=None,
        )

        return match_image

    def load_and_visualize(self, image_name_a, image_name_b, IMAGE_RESIZE=(784, 784)):
        FILE_POSTFIX = f'{config.POSTFIX_DEDODE}_{config.POSTFIX_EUROC}'

        path_a = f"{config.images_dir_path}/{image_name_a}"
        path_b = f"{config.images_dir_path}/{image_name_b}"

        a: ImageSoloData = ImageSoloData(path_a, resize=IMAGE_RESIZE, file_postfix=FILE_POSTFIX)
        a.load_keypoints()

        b: ImageSoloData = ImageSoloData(path_b, resize=IMAGE_RESIZE, file_postfix=FILE_POSTFIX)
        b.load_keypoints()

        pair = ImagePairData(a, b, file_postfix=self.file_postfix)
        pair.load_matches()

        image = self.draw_matches(
            a.image, pair.left_matches,
            b.image, pair.right_matches
        )

        self.plot_image(image)

    """
    Match
    """

    def match_pair(self, pair: ImagePairData):
        a: ImageSoloData = pair.a
        b: ImageSoloData = pair.b

        matches_A, matches_B, batch_ids = self.matcher.match(
            a.keypoints, a.descriptions,
            b.keypoints, b.descriptions,
            normalize=True,
            inv_temp=20,
            threshold=0.1
        )

        pair.left_matches, pair.right_matches = self.matcher.to_pixel_coords(
            matches_A, matches_B,
            a.H, a.W, b.H, b.W
        )

    def match(self):
        a: Optional[ImageSoloData] = None

        IMAGE_RESIZE = (784, 784)
        FILE_POSTFIX = f'{config.POSTFIX_DEDODE}_{config.POSTFIX_EUROC}'

        for index in range(len(self.image_names) - 1):
            path_a = f"{config.images_dir_path}/{self.image_names[index]}"
            path_b = f"{config.images_dir_path}/{self.image_names[index + 1]}"

            if a is None:
                a = ImageSoloData(path_a, resize=IMAGE_RESIZE, file_postfix=FILE_POSTFIX)
                a.load_keypoints()

            b = ImageSoloData(path_b, resize=IMAGE_RESIZE, file_postfix=FILE_POSTFIX)
            b.load_keypoints()

            pair = ImagePairData(a, b, file_postfix=self.file_postfix)

            self.match(pair)
            pair.save_matches()

            a = b


class RoMaMatcher(KeypointMatcher):
    def __init__(self, image_names):
        super().__init__()
        self.image_names = image_names

        from romatch import roma_outdoor
        self.model = roma_outdoor(device=self.device, coarse_res=560, upsample_res=(864, 1152))
        self.model.symmetric = False
        self.H, self.W = self.model.get_output_resolution()

    """
    Utils
    """

    def get_corresponding_points(self, points_in_im1, warp):
        points_in_im2 = []

        for pt in points_in_im1:
            y_im1, x_im1 = pt

            w = warp[y_im1, x_im1]
            A, B = self.model.to_pixel_coordinates(w, self.H, self.W, self.H, self.W)
            x_im2, y_im2 = B

            points_in_im2.append((int(y_im2.item()), int(x_im2.item())))

        return points_in_im2

    @staticmethod
    def get_points_from_certainty(certainty, threshold=0.6, num_points=5):
        # Convert certainty to a binary mask where values are above the threshold
        mask = certainty > threshold

        # Get the coordinates of the points where mask is True
        y_coords, x_coords = np.where(mask)

        # Randomly sample from the points above the threshold
        if len(y_coords) > num_points:
            indices = np.random.choice(len(y_coords), size=num_points, replace=False)
        else:
            indices = np.arange(len(y_coords))  # Take all if not enough points

        points = [(y_coords[i], x_coords[i]) for i in indices]

        return points

    """
    Visualization
    """

    @staticmethod
    def _combine_images(im1: Image, im2: Image):
        combined_images = Image.new('RGB', (im1.width + im2.width, im1.height))
        combined_images.paste(im1, (0, 0))
        combined_images.paste(im2, (im1.width, 0))
        return combined_images

    @staticmethod
    def _random_color_generator():
        color = np.random.randint(0, 256, size=3)
        return tuple(color)

    def match_and_visualize_keypoints(self, points_in_im1, warp, im1: Image, im2: Image):
        combined_images = self._combine_images(im1, im2)
        draw = ImageDraw.Draw(combined_images)

        circle_radius = 10

        for pt in points_in_im1:
            y_im1, x_im1 = pt

            w = warp[y_im1, x_im1]
            A, B = self.model.to_pixel_coordinates(w, self.H, self.W, self.H, self.W)
            x_im2, y_im2 = B

            random_color = self._random_color_generator()

            draw.ellipse(
                (x_im1 - circle_radius, y_im1 - circle_radius, x_im1 + circle_radius, y_im1 + circle_radius),
                outline=random_color,
                width=5
            )

            draw.ellipse(
                (im1.width + x_im2 - circle_radius, y_im2 - circle_radius, im1.width + x_im2 + circle_radius, y_im2 + circle_radius),
                outline=random_color,
                width=5
            )

        return combined_images

    def visualize_keypoints(self, points_in_im1, points_in_im2, im1: Image, im2: Image):
        combined_images = self._combine_images(im1, im2)
        draw = ImageDraw.Draw(combined_images)

        circle_radius = 10

        for pt1, pt2 in zip(points_in_im1, points_in_im2):
            y_im1, x_im1 = pt1
            y_im2, x_im2 = pt2

            random_color = self._random_color_generator()

            draw.ellipse(
                (x_im1 - circle_radius, y_im1 - circle_radius, x_im1 + circle_radius, y_im1 + circle_radius),
                outline=random_color,
                width=5
            )

            draw.ellipse(
                (im1.width + x_im2 - circle_radius, y_im2 - circle_radius, im1.width + x_im2 + circle_radius, y_im2 + circle_radius),
                outline=random_color,
                width=5
            )

        return combined_images

    """
    Match
    """

    def match_pair(self, im_A_path, im_B_path):
        warp, certainty = self.model.match(im_A_path, im_B_path, device=self.device)
        return warp, certainty

    def match(self):
        b: Optional[ImageSoloData] = None

        for index in range(len(self.image_names) - 1):
            path_a = f"{config.images_dir_path}/{self.image_names[index]}"
            path_b = f"{config.images_dir_path}/{self.image_names[index + 1]}"

            warp, certainty = self.match_pair(path_a, path_b)

            a: ImageSoloData = b if b is not None else ImageSoloData(path_a)
            b: ImageSoloData = ImageSoloData(path_b)

            if a.keypoints is None:
                a.load_keypoints()

            b.load_keypoints()

            points_in_im1 = a.keypoints
            points_in_im2 = self.get_corresponding_points(points_in_im1, warp)
