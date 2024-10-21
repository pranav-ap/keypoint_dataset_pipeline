from setuptools import setup, find_packages

setup(
    name='keypoint_dataset_pipeline',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Dependencies can be installed from local paths using the `-e` (editable) flag
        'RoMa @ file:///D:/thesis_code/keypoint_dataset_pipeline/RoMa',
        'DeDoDe @ file:///D:/thesis_code/keypoint_dataset_pipeline/DeDoDe'
    ],
    # Other metadata options...
)
