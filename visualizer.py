import random
import cv2
from ImageData import ImageSoloData, ImagePairData
import numpy as np
from PIL import Image


class Visualizer:
    @staticmethod
    def show_keypoints(path_im, num_points=None):
        im = ImageSoloData(path_im)
        im.load_keypoints()

        num_points = len(im.keypoints_coords) if num_points is None else min(num_points, len(im.keypoints_coords))
        keypoints = random.sample(im.keypoints_coords, num_points)

        image_vis = cv2.drawKeypoints(
            np.array(im.image),
            keypoints,
            None
        )

        image_vis = Image.fromarray(image_vis)
        return image_vis

    @staticmethod
    def show_sparse_matches(path_a, path_b, num_points=10):
        a = ImageSoloData(path_a)
        a.load_keypoints()

        b = ImageSoloData(path_b)
        b.load_keypoints()

        pair = ImagePairData(a, b)
        pair.load_matches()

        return Visualizer.plot_matches(pair, num_points)

    @staticmethod
    def plot_matches(pair: ImagePairData, num_points):
        num_points = len(pair.left_matches_coords) if num_points is None else min(num_points, len(pair.left_matches_coords))
        matches = [cv2.DMatch(idx, idx, 0.) for idx in range(num_points)]

        image_vis = cv2.drawMatches(
            np.array(pair.a.image), pair.left_matches_coords,
            np.array(pair.b.image), pair.right_matches_coords,
            matches,
            outImg=None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        image_vis = Image.fromarray(image_vis)
        return image_vis
