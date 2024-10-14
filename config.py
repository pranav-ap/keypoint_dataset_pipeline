from dataclasses import dataclass


@dataclass
class InferenceConfig:
    images_dir_path: str = ''
    npy_dir_path: str = ''
    csv_path: str = ''

    POSTFIX_DEDODE: str = 'dedode'
    POSTFIX_ROMA: str = 'roma'

    POSTFIX_EUROC: str = 'euroc'
    POSTFIX_KITTI: str = 'kitti'


config = InferenceConfig()

