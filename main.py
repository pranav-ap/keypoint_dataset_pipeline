from data_pipeline import DataPipeline
from rich.console import Console


def main():
    pipeline = DataPipeline()
    pipeline.run()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        console = Console()
        console.print_exception(show_locals=True)
