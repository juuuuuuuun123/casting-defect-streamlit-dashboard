from __future__ import annotations

import argparse
import json

from .config import PROCESSED_DIR, ExperimentConfig, ensure_project_dirs
from .data import build_metadata, save_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw casting images and create metadata with holdout and CV folds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-size", type=float, default=0.15)
    parser.add_argument("--num-folds", type=int, default=5)
    args = parser.parse_args()

    ensure_project_dirs()
    config = ExperimentConfig(seed=args.seed, holdout_size=args.holdout_size, num_folds=args.num_folds)
    metadata = build_metadata(config=config)
    summary = save_summary(metadata)
    (PROCESSED_DIR / "label_mapping.json").write_text(
        json.dumps({"ok_front": 0, "def_front": 1}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
