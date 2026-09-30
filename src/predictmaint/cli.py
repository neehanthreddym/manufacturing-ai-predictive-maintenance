import argparse
import json
from pathlib import Path

from predictmaint.config import load_config
from predictmaint.data.download import download_data
from predictmaint.data.preprocess import prepare_data
from predictmaint.logging_config import configure_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproducible manufacturing AI pipeline")
    parser.add_argument("stage", choices=["data", "prepare", "train", "evaluate"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    configure_logging()
    config = load_config(args.config)
    if args.stage == "data":
        download_data(config.data.raw_path, args.force)
    elif args.stage == "prepare":
        prepare_data(config)
    elif args.stage == "train":
        from predictmaint.modeling.train import train

        train(config, args.quick)
    else:
        # Training evaluates the held-out set once. This stage reads the stored report.
        path = Path("reports/final_metrics.json")
        if not path.exists():
            raise FileNotFoundError("Run make train first; evaluation is saved during training")
        print(json.dumps(json.loads(path.read_text()), indent=2))


if __name__ == "__main__":
    main()
