import random
import cv2
from config import config
from ImageData import ImageSoloData, ImagePairData
import numpy as np
from PIL import Image


class Visualizer:
    @staticmethod
    def show_keypoints(path_im, num_points=None):
        im = ImageSoloData(path_im, resize=config.IMAGE_RESIZE)
        im.load_keypoints()

        width, height = im.image.size
        image = np.array(im.image)

        keypoints = [
            cv2.KeyPoint((x + 1) * (width / 2), (y + 1) * (height / 2), 1.)
            for x, y in im.keypoints.squeeze(0)
        ]

        num_points = len(keypoints) if num_points is None else min(num_points, len(keypoints))
        keypoints = random.sample(keypoints, num_points)

        image_vis = cv2.drawKeypoints(
            image,
            keypoints,
            None
        )

        image_vis = Image.fromarray(image_vis)
        return image_vis

    @staticmethod
    def show_sparse_matches(path_a, path_b, num_points=10):
        a = ImageSoloData(path_a, resize=config.IMAGE_RESIZE)
        a.load_keypoints()

        b = ImageSoloData(path_b, resize=config.IMAGE_RESIZE)
        b.load_keypoints()

        pair = ImagePairData(a, b)
        pair.load_matches()

        a, b = np.array(a.image), np.array(b.image)

        left_matches = [
            cv2.KeyPoint(x, y, 1.)
            for x, y in pair.left_matches
        ]

        right_matches = [
            cv2.KeyPoint(x, y, 1.)
            for x, y in pair.right_matches
        ]

        num_points = len(left_matches) if num_points is None else min(num_points, len(left_matches))
        matches = [cv2.DMatch(idx, idx, 0.) for idx in range(num_points)]

        image_vis = cv2.drawMatches(
            a, left_matches,
            b, right_matches,
            matches,
            outImg=None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        image_vis = Image.fromarray(image_vis)

        return image_vis
