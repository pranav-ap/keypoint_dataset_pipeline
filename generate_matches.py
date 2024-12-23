from src import DataPipeline
from config import config


def main():
    pipeline = DataPipeline()
    # pipeline.run()

    if config.task.extract_from_all_tracks:
        pipeline.run_list()
    else:
        pipeline.run()


if __name__ == '__main__':
    main()
