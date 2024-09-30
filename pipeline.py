from dataclasses import dataclass
import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path


@dataclass
class InferenceConfig:
    data_path = "assets/"
    image_data_path = "/kaggle/working/"


config = InferenceConfig()


class ImageData:
    def __init__(self, image_path, keypoints, descriptions):
        self.image_path: str = image_path
        self.image_name: str = Path(self.image_path).stem

        image, W, H = self._read_image_and_size(image_path)

        self.image: Image.Image = image
        self.W: int = W
        self.H: int = H

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

    @staticmethod
    def _read_image_and_size(image_path):
        image: Image.Image = Image.open(image_path)
        W, H = image.size
        return image, W, H

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
            self.right_matches = np.load(right_matches_filepath)

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


def get_best_device(verbose=False):
    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if verbose:
        print(f"Fastest device found is: {device}")

    return device


class DatasetPipeline:
    def __init__(self):
        from DeDoDe import dedode_detector_L, dedode_descriptor_G
        from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher

        self.detector = dedode_detector_L(weights=None)
        self.descriptor = dedode_descriptor_G(weights=None, dinov2_weights=None)
        self.matcher = DualSoftMaxMatcher()

        self.deblurer = None

    """
    UTILS
    """

    @staticmethod
    def _get_image_paths():
        # im_A_path = "assets/im_A.jpg"
        # im_B_path = "assets/im_B.jpg"
        im_A_path = "assets/1403715284512143104.jpg"
        im_B_path = "assets/1403715285812143104.jpg"

        return [im_A_path, im_B_path]

    """
    Detect, Describe, Match
    """

    def detect(self, image_path, H=784, W=784, device=get_best_device()):
        pil_im = Image.open(image_path).resize((W, H))

        # Convert grayscale images to RGB
        if pil_im.mode != 'RGB':
            pil_im = pil_im.convert('RGB')

        standard_im = np.array(pil_im) / 255.0

        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Check if the image is grayscale (2D array)
        if standard_im.ndim == 2:  # Grayscale
            # Add a channel dimension
            standard_im = standard_im[None, :, :]  # Shape becomes (1, H, W)
        else:  # RGB
            # Permute dimensions for RGB images
            standard_im = np.transpose(standard_im, (2, 0, 1))  # Shape becomes (3, H, W)

        # Apply normalization
        standard_im = normalizer(torch.from_numpy(standard_im)).float().to(device)[None]  # Add batch dimension

        batch = {"image": standard_im}
        detections = self.detector.detect(batch, num_keypoints=10_000)
        keypoints, confidences = detections["keypoints"], detections["confidence"]

        return keypoints, confidences

    def describe(self, image_path, keypoints, H=784, W=784, device=get_best_device()):
        pil_im = Image.open(image_path).resize((W, H))

        # Convert grayscale images to RGB
        if pil_im.mode != 'RGB':
            pil_im = pil_im.convert('RGB')

        standard_im = np.array(pil_im) / 255.0

        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Check if the image is grayscale (2D array)
        if standard_im.ndim == 2:  # Grayscale
            # Add a channel dimension
            standard_im = standard_im[None, :, :]  # Shape becomes (1, H, W)
        else:  # RGB
            # Permute dimensions for RGB images
            standard_im = np.transpose(standard_im, (2, 0, 1))  # Shape becomes (3, H, W)

        # Apply normalization
        standard_im = normalizer(torch.from_numpy(standard_im)).float().to(device)[None]  # Add batch dimension

        batch = {"image": standard_im}
        descriptions = self.descriptor.describe_keypoints(batch, keypoints)
        descriptions = descriptions["descriptions"]

        return descriptions

    def detect_describe(self, image_path) -> ImageData:
        keypoints, confidences = self.detect(image_path)
        descriptions = self.describe(image_path, keypoints)

        image_data = ImageData(image_path, keypoints, descriptions)
        return image_data

    def match(self, a: ImageData, b: ImageData):
        matches_A, matches_B, batch_ids = self.matcher.match(
            a.keypoints, a.descriptions,
            b.keypoints, b.descriptions,
            normalize=True,
            inv_temp=20,
            threshold=0.1
        )

        a.left_matches, b.right_matches = self.matcher.to_pixel_coords(matches_A, matches_B, a.H, a.W, b.H, b.W)

    def detect_describe_match(self):
        image_path_list = self._get_image_paths()

        a: ImageData | None = None
        b: ImageData | None = None

        for index in range(len(image_path_list) - 1):
            path_a = image_path_list[index]
            path_b = image_path_list[index + 1]

            # Detect & Describe

            a = b if b is not None else self.detect_describe(path_a)
            b = self.detect_describe(path_b)

            a.save_keypoints()

            # Match

            self.match(a, b)

            a.save_left_matches()
            b.save_right_matches()

        # Save Last Image Data
        b.save_keypoints()
        b.save_left_matches()

    """
    RUN
    """

    def run(self):
        pass

    """
    Checkouts
    """

    def checkout_debluring(self):
        pass

    def checkout_matching(self, path_a, path_b):
        a = self.detect_describe(path_a)
        b = self.detect_describe(path_b)

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
    pipeline.detect_describe_match()


if __name__ == '__main__':
    main()
