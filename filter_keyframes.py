import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import config
from utils import logger


def read_calib_json():
    file_path = ''
    with open(file_path, 'r') as file:
        data = json.load(file) 
        


def read_imu_csv():
    imu_path = config.paths.basalt.imu_csv
    imu = pd.read_csv(imu_path, header=0, names=('timestamp', 'w_x', 'w_y', 'w_z', 'a_x', 'a_y', 'a_z'))
    imu['timestamp'] = pd.to_datetime(imu['timestamp'], unit='ns')
    imu = imu.sort_values(by='timestamp').reset_index(drop=True)

    return imu


def read_images_csv():
    files_path = config.paths.basalt.images_csv
    files = pd.read_csv(files_path, header=0, names=('timestamp', 'filename'))
    files['timestamp'] = pd.to_datetime(files['timestamp'], unit='ns')
    files = files.sort_values(by='timestamp').reset_index(drop=True)

    return files


def align_rows(track, cam, imu):
    logger.info(f'Align Rows')

    files = read_images_csv()
    # aligned_df stores files df timestamp
    aligned_df = pd.merge_asof(
        files, imu,
        on="timestamp",
        direction="backward",
        suffixes=('', '_imu')
    )

    aligned_df = aligned_df.dropna()
    aligned_df.to_csv(config.paths.basalt.aligned_csv, index=False, header=True)

    logger.info(f"Stats for aligned DataFrame ({track}, {cam}):")
    logger.info(f"Shape of DataFrame: {aligned_df.shape}")

    return aligned_df


def filter_rows(track, cam, aligned_df):
    logger.info(f'Filter Keyframes')

    aligned_df['dt'] = aligned_df['timestamp'].diff().dt.total_seconds()

    # Integrating acceleration to get velocity (assuming initial velocity = 0)
    aligned_df['velocity_x'] = (aligned_df['a_x'] * aligned_df['dt']).cumsum()
    aligned_df['velocity_y'] = (aligned_df['a_y'] * aligned_df['dt']).cumsum()
    aligned_df['velocity_z'] = (aligned_df['a_z'] * aligned_df['dt']).cumsum()

    # Calculate position by integrating velocity (assuming initial position = 0)
    aligned_df['position_x'] = (aligned_df['velocity_x'] * aligned_df['dt']).cumsum()
    aligned_df['position_y'] = (aligned_df['velocity_y'] * aligned_df['dt']).cumsum()
    aligned_df['position_z'] = (aligned_df['velocity_z'] * aligned_df['dt']).cumsum()

    # Euclidean  distance between consecutive rows
    aligned_df['displacement'] = np.sqrt(
        (aligned_df['position_x'].diff()) ** 2 +
        (aligned_df['position_y'].diff()) ** 2 +
        (aligned_df['position_z'].diff()) ** 2
    )

    logger.info(f"Min displacement: {aligned_df['displacement'].min()}")
    logger.info(f"Max displacement: {aligned_df['displacement'].max()}")

    # Define a threshold
    displacement_threshold = 0.2  # in meters, adjust as necessary

    # keyframes_df = aligned_df[aligned_df['displacement'] > displacement_threshold]

    # Find the next row that satisfies the threshold for each row
    keyframes_indices = []
    i = 0

    while i < len(aligned_df):
        keyframes_indices.append(i)
        next_row = aligned_df['displacement'][i + 1:].gt(displacement_threshold).idxmax()

        if aligned_df['displacement'][next_row] <= displacement_threshold or next_row <= i:
            break

        i = next_row

    keyframes_df = aligned_df.iloc[keyframes_indices].reset_index(drop=True)

    logger.info(f"Stats for filtered DataFrame ({track}, {cam}):")
    logger.info(f"Shape of DataFrame: {keyframes_df.shape}")

    keyframes_df.to_csv(config.paths.basalt.keyframes_csv, index=False, header=True)


def align_and_filter_rows():
    for track in config.task.tracks:
        config.task.track = track

        logger.info(f'Align & Filter {track}')

        imu = read_imu_csv()

        for cam in tqdm(config.task.cams, total=len(config.task.cams), desc=f'Filtering keyframes : {track}',
                        ncols=100):
            config.task.cam = cam

            aligned_df = align_rows(track, cam, imu)
            filter_rows(track, cam, aligned_df)


def main():
    align_and_filter_rows()


if __name__ == '__main__':
    main()
