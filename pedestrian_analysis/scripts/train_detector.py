"""Optional local fine-tuning scaffold for pedestrian-only detector training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8l.pt", help="Pretrained checkpoint to fine-tune locally.")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Use high resolution for small pedestrians.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="pedestrian_analysis/outputs/training")
    parser.add_argument("--name", default="pedestrian_small_target_ft")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Ultralytics is required for local fine-tuning.") from exc

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
        close_mosaic=10,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        scale=0.3,
    )


if __name__ == "__main__":
    main()
