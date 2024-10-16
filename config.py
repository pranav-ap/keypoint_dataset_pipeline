from dataclasses import dataclass


@dataclass
class InferenceConfig:
    num_keypoints_to_detect: int = 10_000
    IMAGE_RESIZE: tuple[int, int] = (784, 784)

    images_dir_path: str = ''
    npy_dir_path: str = ''
    csv_path: str = ''

    POSTFIX_DETECTOR_MODEL: str = ''
    POSTFIX_MATCHER_MODEL: str = ''
    POSTFIX_DATASET: str = ''

    """
    INTERNAL USE
    """

    device = 'cpu'


config = InferenceConfig()
