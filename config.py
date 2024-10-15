from dataclasses import dataclass


@dataclass
class InferenceConfig:
    num_keypoints_to_detect: int = 1000
    IMAGE_RESIZE: tuple[int, int] = (784, 784)

    images_dir_path: str = ''
    npy_dir_path: str = ''
    csv_path: str = ''

    POSTFIX_MODEL: str = 'dedode'
    POSTFIX_DATASET: str = 'euroc'

    """
    INTERNAL USE
    """

    device = None
    POSTFIX_FILE: str = ''


config = InferenceConfig()

