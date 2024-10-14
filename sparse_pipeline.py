from config import config
from utils.logger import logger
from utils import get_best_device
from ImageData import ImageSoloData, ImagePairData
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from typing import Optional
from abc import ABC, abstractmethod


class KeypointDetector(ABC):
    def __init__(self, num_keypoints):
        self.device = get_best_device()
        self.num_keypoints = num_keypoints

    @abstractmethod
    def extract_keypoints(self):
        pass


class KeypointMatcher(ABC):
    def __init__(self, file_postfix):
        self.device = get_best_device()
        self.file_postfix = file_postfix

    @abstractmethod
    def match(self):
        pass

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

    def load_and_visualize(self, image_name_a, image_name_b, IMAGE_RESIZE = (784, 784)):
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


class DeDoDeDetector(KeypointDetector):
    def __init__(self, image_names, num_keypoints=1000):
        super().__init__(num_keypoints)

        self.image_names = image_names

        from DeDoDe import dedode_detector_L, dedode_descriptor_G
        self.detector = dedode_detector_L(weights=None)
        self.descriptor = dedode_descriptor_G(weights=None, dinov2_weights=None)

        self.normalizer = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    """
    Utils
    """

    def preprocess_image(self, image_data: ImageSoloData):
        # Convert grayscale to RGB if necessary
        if image_data.image.mode != 'RGB':
            image_data.image = image_data.image.convert('RGB')

        standard_im = np.array(image_data.image) / 255.0

        # Convert grayscale to 3-channel if needed
        if standard_im.ndim == 2:
            standard_im = np.stack([standard_im] * 3, axis=0)  # (3, H, W)
        else:
            standard_im = np.transpose(standard_im, (2, 0, 1))  # (3, H, W)

        standard_im = self.normalizer(torch.from_numpy(standard_im)).float()

        return standard_im

    """
    Detect, Describe, Match
    """

    def detect_describe(self, image_data: ImageSoloData):
        """
        Returns an Image Data object that contains keypoints and descriptors
        """

        standard_im = self.preprocess_image(image_data).to(self.device)[None]
        batch = {"image": standard_im}

        detections = self.detector.detect(batch, self.num_keypoints)
        image_data.keypoints = detections["keypoints"]
        image_data.confidences = detections["confidence"]

        descriptions = self.descriptor.describe_keypoints(batch, image_data.keypoints)
        image_data.descriptions = descriptions["descriptions"]

    def extract_keypoints(self):
        a: Optional[ImageSoloData] = None
        b: Optional[ImageSoloData] = None

        IMAGE_RESIZE = (784, 784)
        FILE_POSTFIX = f'{config.POSTFIX_DEDODE}_{config.POSTFIX_EUROC}'

        for index in range(len(self.image_names) - 1):
            path_a = f"{config.images_dir_path}/{self.image_names[index]}"
            path_b = f"{config.images_dir_path}/{self.image_names[index + 1]}"

            if a is None:
                a = ImageSoloData(path_a, resize=IMAGE_RESIZE, file_postfix=FILE_POSTFIX)
                self.detect_describe(a)

            b = ImageSoloData(path_b, resize=IMAGE_RESIZE, file_postfix=FILE_POSTFIX)
            self.detect_describe(b)

            a.save_keypoints()
            a = b

        if b:
            b.save_keypoints()


class DeDoDeMatcher(KeypointMatcher):
    def __init__(self, image_names):
        super().__init__()
        self.image_names = image_names

        from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
        self.matcher = DualSoftMaxMatcher()

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


class SparsePipelineRunner:
    def __init__(self, pipeline):
        self.pipeline = pipeline

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
        # image_names = image_names[:5]

        self.detect_describe_match(image_names)
        self.load_and_visualize(image_names)


def main():
    pipeline = DeDoDePipeline()

    pipeline_runner = SparsePipelineRunner(pipeline)
    pipeline_runner.run()


if __name__ == '__main__':
    main()
