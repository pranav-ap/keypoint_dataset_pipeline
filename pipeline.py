from dataclasses import dataclass
import os
import numpy as np
from PIL import Image
from pathlib import Path
from utils.visualize import read_image_and_size


@dataclass
class InferenceConfig:
    data_path = "assets/"
    image_data_path = "image_data/"


config = InferenceConfig()


class ImageData:
    def __init__(self, path, keypoints, descriptions):
        self.image_path: str = path
        self.image_name: str = Path(self.image_path).stem
        self.image: Image.Image = Image.open(path)
        self.W: int = self.image.width
        self.H: int = self.image.height

        """
        Example Sizes
        
        keypoints_A : torch.Size([1, 10000, 2])
        descriptions_A : torch.Size([1, 10000, 256])
        matches_A : torch.Size([2626, 2])
        """

        self.keypoints = keypoints
        self.descriptions = descriptions

        self.left_matches = None
        self.right_matches = None

    def load_keypoints(self):
        keypoints_filepath = os.path.join(config.image_data_path, f"{self.image_name}_keypoints.npy")
        assert os.path.exists(keypoints_filepath)
        self.keypoints = np.load(keypoints_filepath)

        descriptions_filepath = os.path.join(config.image_data_path, f"{self.image_name}_descriptions.npy")
        assert os.path.exists(descriptions_filepath)
        self.descriptions = np.load(descriptions_filepath)

    def load_matches(self):
        left_matches_filepath = os.path.join(config.image_data_path, f"{self.image_name}_left_matches.npy")
        if os.path.exists(left_matches_filepath):
            self.left_matches = np.load(left_matches_filepath)

        right_matches_filepath = os.path.join(config.image_data_path, f"{self.image_name}_right_matches.npy")
        if os.path.exists(right_matches_filepath):
            self.left_matches = np.load(right_matches_filepath)

    def save_keypoints(self):
        keypoints_np = self.keypoints.cpu().numpy()
        descriptions_np = self.descriptions.cpu().numpy()

        np.save(os.path.join(config.image_data_path, f"{self.image_name}_keypoints.npy"), keypoints_np)
        np.save(os.path.join(config.image_data_path, f"{self.image_name}_descriptions.npy"), descriptions_np)

    def save_left_matches(self):
        left_matches_np = self.left_matches.cpu().numpy()
        np.save(os.path.join(config.image_data_path, f"{self.image_name}_left_matches.npy"), left_matches_np)

    def save_right_matches(self):
        right_matches_np = self.right_matches.cpu().numpy()
        np.save(os.path.join(config.image_data_path, f"{self.image_name}_right_matches.npy"), right_matches_np)


class DatasetPipeline:
    def __init__(self):
        from DeDoDe import dedode_detector_L, dedode_descriptor_G
        from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher

        self.detector = dedode_detector_L(weights=None)
        self.descriptor = dedode_descriptor_G(weights=None, dinov2_weights=None)
        self.matcher = DualSoftMaxMatcher()

    """
    UTILS
    """

    @staticmethod
    def _get_image_paths():
        im_A_path = "assets/im_A.jpg"
        im_B_path = "assets/im_B.jpg"

        return [im_A_path, im_B_path]

    """
    Detect, Describe, Match
    """

    def detect(self, image_path):
        detections = self.detector.detect_from_path(image_path, num_keypoints=10_000)
        keypoints, confidences = detections["keypoints"], detections["confidence"]
        return keypoints, confidences

    def describe(self, image_path, keypoints):
        descriptions = self.descriptor.describe_keypoints_from_path(
            image_path,
            keypoints
        )

        descriptions = descriptions["descriptions"]

        return descriptions

    def detect_and_describe(self, image_path):
        keypoints, confidences = self.detect(image_path)
        descriptions = self.describe(image_path, keypoints)

        image_data = ImageData(image_path, keypoints, descriptions)
        return image_data

    def match(self, a, b):
        matches_A, matches_B, batch_ids = self.matcher.match(
            a.keypoints, a.descriptions,
            b.keypoints, b.descriptions,
            normalize=True,
            inv_temp=20,
            threshold=0.1
        )

        a.left_matches, b.right_matches = self.matcher.to_pixel_coords(matches_A, matches_B, a.H, a.W, b.H, b.W)

    """
    RUN
    """

    def run(self):
        image_path_list = self._get_image_paths()
        a, b = None, None

        for index in range(len(image_path_list) - 1):
            path_a = image_path_list[index]
            path_b = image_path_list[index + 1]

            # Detect & Describe

            a = b if b is not None else self.detect_and_describe(path_a)
            b = self.detect_and_describe(path_b)

            a.save_keypoints()

            # Match

            self.match(a, b)

            a.save_left_matches()
            b.save_right_matches()

        # Save Last Image Data
        b.save_keypoints()
        b.save_left_matches()

    """
    Checkouts
    """

    def checkout_matching(self, path_a, path_b):
        a = self.detect_and_describe(path_a)
        b = self.detect_and_describe(path_b)

        self.match(a, b)

        from utils.visualize import draw_matches
        match_image = draw_matches(
            a.image, a.left_matches,
            b.image, b.right_matches
        )

        from utils.visualize import plot_single
        plot_single(match_image)


def main():
    pipeline = DatasetPipeline()
    pipeline.run()


if __name__ == '__main__':
    main()
