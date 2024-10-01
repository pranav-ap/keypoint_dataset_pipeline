from dataclasses import dataclass
import os
from pathlib import Path
from utils import *


@dataclass
class InferenceConfig:
    data_path = "assets/"
    image_data_path = "/kaggle/working/"


config = InferenceConfig()


class ImageData:
    def __init__(self, image_path, resize=None):
        self.image_path: str = image_path
        self.image_name: str = Path(self.image_path).stem

        self.image: Image.Image = Image.open(image_path)

        if resize:
            self.image = self.image.resize(resize)

        W, H = self.image.size
        self.W: int = W
        self.H: int = H

        """
        Example Sizes
        
        keypoints_A : torch.Size([1, 10000, 2])
        descriptions_A : torch.Size([1, 10000, 256])
        matches_A : torch.Size([2626, 2])
        """

        self.keypoints = None
        self.confidences = None
        self.descriptions = None

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


class KeypointDatasetPipeline:
    def __init__(self):
        from DeDoDe import dedode_detector_L, dedode_descriptor_G
        from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher

        self.detector = dedode_detector_L(weights=None)
        self.descriptor = dedode_descriptor_G(weights=None, dinov2_weights=None)
        self.matcher = DualSoftMaxMatcher()

        self.normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.device = get_best_device()

        self.deblurer = None

    """
    Detect, Describe, Match
    """

    def preprocess_image(self, image_data: ImageData):
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

    def detect(self, image_data: ImageData, standard_im):
        batch = {"image": standard_im}
        detections = self.detector.detect(batch, num_keypoints=10_000)

        image_data.keypoints = detections["keypoints"]
        image_data.confidences = detections["confidence"]

    def describe(self, image_data: ImageData, standard_im):
        batch = {"image": standard_im}
        descriptions = self.descriptor.describe_keypoints(batch, image_data.keypoints)

        image_data.descriptions = descriptions["descriptions"]

    def detect_describe(self, image_path) -> ImageData:
        image_data = ImageData(image_path, resize=(784, 784))
        standard_im = self.preprocess_image(image_path).to(self.device)[None]  # Add batch dimension

        self.detect(image_data, standard_im)
        self.describe(image_data, standard_im)

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

    def detect_describe_match(self, folder_path, image_names):
        a: ImageData | None = None
        b: ImageData | None = None

        for index in range(len(image_names) - 1):
            path_a = f"{folder_path}/{image_names[index]}"
            path_b = f"{folder_path}/{image_names[index + 1]}"

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
    Prepare Image Names
    """

    @staticmethod
    def _get_folder_and_image_names():
        # EUROC V1_01_easy
        folder_path = "D:/thesis_code/data/V1_01_easy/mav0/cam0/data"
        csv_path = "D:/thesis_code/data/V1_01_easy/mav0/cam0/data.csv"

        import pandas as pd
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp [ns]'], unit='ns')
        df = df.sort_values(by='timestamp')

        image_names = df['filename'].tolist()

        return folder_path, image_names

    """
    RUN
    """

    def run(self):
        folder_path, image_names = self._get_folder_and_image_names()
        self.detect_describe_match(folder_path, image_names)

    """
    Checkout
    """

    def checkout_matching(self, path_a, path_b):
        a: ImageData = self.detect_describe(path_a)
        b: ImageData = self.detect_describe(path_b)

        self.match(a, b)

        from utils import draw_matches
        match_image = draw_matches(
            a.image, a.left_matches,
            b.image, b.right_matches,
            count=15
        )

        from utils import plot_single
        plot_single(match_image)


def main():
    pipeline = KeypointDatasetPipeline()

    # im_A_path = "assets/im_A.jpg"
    # im_B_path = "assets/im_B.jpg"

    im_A_path = "assets/1403715284512143104.jpg"
    im_B_path = "assets/1403715285812143104.jpg"

    pipeline.checkout_matching(im_A_path, im_B_path)


if __name__ == '__main__':
    main()
