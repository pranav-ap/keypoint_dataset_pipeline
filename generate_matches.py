from src import DataPipeline
from config import config


def main():
    print(config)
    pipeline = DataPipeline()

    if config.task.only_missing:
        pipeline.run_missing_list()
    elif config.task.extract_from_all_tracks:
        pipeline.run_list()
    else:
        pipeline.run()


if __name__ == '__main__':
    main()
