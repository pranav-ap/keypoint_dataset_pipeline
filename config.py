from dataclasses import dataclass


@dataclass
class InferenceConfig:
    num_keypoints_to_detect: int = 10_000
    IMAGE_RESIZE: tuple[int, int] = (784, 784)

    images_dir_path: str = ''
    npy_dir_path: str = ''
    csv_path: str = ''

    POSTFIX_MODEL: str = ''
    POSTFIX_DATASET: str = ''

    circle_radius: int = 8

    """
    INTERNAL USE
    """

    device = None


config = InferenceConfig()
