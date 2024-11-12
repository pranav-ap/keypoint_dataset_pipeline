from datetime import timedelta

import pandas as pd
from tqdm import tqdm

from config import config


class IMUFilter:
    @staticmethod
    def average_last_n_imu_rows(imu_df, image_df, N=10_000):
        averaged_imu_data = []

        for _, image_row in tqdm(image_df.iterrows(), total=len(image_df), desc="Processing IMU data (N rows)"):
            image_timestamp = image_row["timestamp"]
            filename = image_row["filename"]

            # Last N IMU rows before current image timestamp
            mask = imu_df["timestamp"] <= image_timestamp
            imu_before = imu_df[mask].tail(N)

            # Calculate the time span covered by these N rows
            time_span_seconds = 0
            if len(imu_before) > 1:
                time_span_seconds = imu_before["timestamp"].iloc[-1] - imu_before["timestamp"].iloc[0]

            averaged_imu_data.append({
                "timestamp": image_timestamp,
                "filename": filename,
                "avg_accel_x": imu_before["accel_x"].mean(),
                "avg_accel_y": imu_before["accel_y"].mean(),
                "avg_accel_z": imu_before["accel_z"].mean(),
                "avg_gyro_x": imu_before["gyro_x"].mean(),
                "avg_gyro_y": imu_before["gyro_y"].mean(),
                "avg_gyro_z": imu_before["gyro_z"].mean(),
                "time_span_seconds": time_span_seconds
            })

        return pd.DataFrame(averaged_imu_data)

    @staticmethod
    def average_last_t_seconds_imu(imu_df, image_df, T_seconds=2):
        averaged_imu_data = []

        for _, image_row in tqdm(image_df.iterrows(), total=len(image_df), desc="Processing IMU data (T seconds)"):
            image_timestamp = image_row["timestamp"]
            filename = image_row["filename"]

            # Start time is T seconds before the current image timestamp
            start_time = image_timestamp - timedelta(seconds=T_seconds)

            # Get IMU rows within [start_time, image_timestamp]
            imu_before = imu_df[(imu_df["timestamp"] > start_time) & (imu_df["timestamp"] <= image_timestamp)]

            # Calculate the time span covered by these rows
            time_span_seconds = 0
            if len(imu_before) > 1:
                time_span_seconds = (imu_before["timestamp"].iloc[-1] - imu_before["timestamp"].iloc[0]).total_seconds()

            averaged_imu_data.append({
                "timestamp": image_timestamp,
                "filename": filename,
                "avg_accel_x": imu_before["accel_x"].mean(),
                "avg_accel_y": imu_before["accel_y"].mean(),
                "avg_accel_z": imu_before["accel_z"].mean(),
                "avg_gyro_x": imu_before["gyro_x"].mean(),
                "avg_gyro_y": imu_before["gyro_y"].mean(),
                "avg_gyro_z": imu_before["gyro_z"].mean(),
                "time_span_seconds": time_span_seconds
            })

        return pd.DataFrame(averaged_imu_data)

    def process_imu(self):
        filepath_imu = config.paths[config.task.name].imu_csv
        df_imu = pd.read_csv(filepath_imu, header=0,
                             names=('timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'))
        df_imu['timestamp'] = pd.to_datetime(df_imu['timestamp'], unit='ns')

        filepath_images = config.paths[config.task.name].images_csv
        df_images = pd.read_csv(filepath_images, header=0, names=('timestamp', 'filename'))
        df_images['timestamp'] = pd.to_datetime(df_images['timestamp'], unit='ns')
        df_images = df_images.sort_values(by='timestamp')
        df_images['filename'] = df_images['filename'].str.replace(".png", "", regex=False)

        # N = 10_000
        # df = self.average_last_n_imu_rows(df_imu, df_images, N)

        T_seconds = config.imu_filter.lookback_seconds
        df = self.average_last_t_seconds_imu(df_imu, df_images, T_seconds)

        return df

    def extract(self):
        df = self.process_imu()
        imu_filepath = f'{config.paths[config.task.name].output}/imu_data.csv'
        df.to_csv(imu_filepath, index=False)
