from utils import get_best_device
import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from romatch.utils.utils import tensor_to_pil
from romatch import roma_outdoor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class DensePipeline:
    def __init__(self):
        device = get_best_device()
        self.device = device

        self.model = roma_outdoor(device=device, coarse_res=560, upsample_res=(864, 1152))
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
    def _show_combined_images(im1: Image, im2: Image):
        combined_images = Image.new('RGB', (im1.width + im2.width, im1.height))
        combined_images.paste(im1, (0, 0))
        combined_images.paste(im2, (im1.width, 0))
        return combined_images

    @staticmethod
    def _random_color_generator():
        color = np.random.randint(0, 256, size=3)
        return tuple(color)

    def match_and_visualize_keypoints(self, points_in_im1, warp, im1: Image, im2: Image):
        combined_images = self._show_combined_images(im1, im2)
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
        combined_images = self._show_combined_images(im1, im2)
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

    def match(self, image_names):
        b: Optional[ImageData] = None

        for index in range(len(image_names) - 1):
            path_a = f"{config.images_dir_path}/{image_names[index]}"
            path_b = f"{config.images_dir_path}/{image_names[index + 1]}"

            warp, certainty = self.match_pair(path_a, path_b)

            a: ImageData = b if b is not None else ImageData(path_a)
            b: ImageData = ImageData(path_b)

            if a.keypoints is None:
                a.load_keypoints()

            b.load_keypoints()

            points_in_im1 = a.keypoints
            points_in_im2 = self.get_corresponding_points(points_in_im1, warp)


class DensePipelineRunner:
    def __init__(self, pipeline: DensePipeline):
        self.pipeline = pipeline

    """
    Run
    """

    @staticmethod
    def _get_sorted_image_names():
        df = pd.read_csv(config.csv_path)
        df['timestamp'] = pd.to_datetime(df['#timestamp [ns]'], unit='ns')
        df = df.sort_values(by='timestamp')
        image_names = df['filename'].tolist()

        return image_names

    def run(self):
        config.images_dir_path = "/kaggle/input/euroc-v1-01-easy/V1_01_easy/data"
        config.csv_path = "/kaggle/input/euroc-v1-01-easy/V1_01_easy/data.csv"
        config.npy_dir_path = "/kaggle/working/euroc-v1-01-easy/npy_files"

        image_names = self._get_sorted_image_names()
        image_names = image_names[:3]


def main():
    pipeline = DensePipeline()

    pipeline_runner = DensePipelineRunner(pipeline)
    pipeline_runner.run()


if __name__ == '__main__':
    main()
