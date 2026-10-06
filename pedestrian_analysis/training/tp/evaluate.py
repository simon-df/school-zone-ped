"""Evaluate TP checkpoints (and the constant-velocity baseline) on a dataset.

Every checkpoint is loaded through the *same* code path as the TP Analysis
tab (:meth:`pipeline.tp_model_registry.TPModelRegistry.create_adapter`, which
also applies the sidecar ``.json`` architecture settings), so a checkpoint
that evaluates here also works in the UI. The constant-velocity baseline
(:class:`pipeline.tp_adapters.DummyTPAdapter`) is always included.

Examples (run from ``pedestrian_analysis/``)::

    # ETH/UCY test scene of a leave-one-out split
    python -m training.tp.evaluate --dataset ethucy --test-scene eth \\
        --checkpoint lstm=outputs/tp_training/social_lstm_ethucy_eth/best.pt

    # zero-shot on your own drone CSVs
    python -m training.tp.evaluate --dataset own_csv --csv data/trajectories/ \\
        --checkpoint pretrained=outputs/tp_training/social_lstm_ethucy_eth/best.pt

Reports ADE / FDE (first mode) and minADE@K / minFDE@K in metres, prints a
table and writes ``<output>.json`` + ``<output>.csv``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pipeline.tp_adapters import BaseTPAdapter, DummyTPAdapter  # noqa: E402
from pipeline.tp_model_registry import (  # noqa: E402
    TPModelRegistry,
    checkpoint_settings_warnings,
    load_checkpoint_metadata,
)
from training.tp.datasets import (  # noqa: E402
    DEFAULT_ETHUCY_ROOT,
    ETHUCY_SCENES,
    BaseTrajectoryDataset,
    ETHUCYDataset,
    OwnCSVDataset,
    TrajectoryWindow,
)
from training.tp.metrics import MetricAccumulator  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "tp_training"
REPORT_COLUMNS: tuple[str, ...] = ("label", "model", "ade", "fde", "min_ade", "min_fde", "num_agents", "num_windows", "checkpoint")


def evaluate_adapter(
    adapter: BaseTPAdapter,
    windows: Iterable[TrajectoryWindow],
    num_modes: int,
    pred_len: int,
) -> dict[str, float]:
    """Run *adapter* on every window (all co-present agents together) and compute metrics."""
    acc = MetricAccumulator()
    num_windows = 0
    for window in windows:
        pred = adapter.predict(window.obs, num_modes=num_modes, pred_len=pred_len)
        acc.update(pred, window.fut)
        num_windows += 1
    result = acc.compute()
    result["num_windows"] = num_windows
    return result


def run_evaluation(
    windows: Sequence[TrajectoryWindow],
    checkpoints: Sequence[tuple[str, str, Optional[str]]],
    num_modes: int,
    pred_len: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Evaluate the constant-velocity baseline plus each ``(label, model, checkpoint_path)``."""
    rows: list[dict[str, Any]] = []
    baseline = evaluate_adapter(DummyTPAdapter(pred_len=pred_len), windows, num_modes, pred_len)
    rows.append({"label": "constant_velocity", "model": "dummy", "checkpoint": "", **baseline})
    for label, model, path in checkpoints:
        adapter = TPModelRegistry.create_adapter(model, checkpoint_path=path, pred_len=pred_len, seed=seed)
        metrics = evaluate_adapter(adapter, windows, num_modes, pred_len)
        rows.append({"label": label, "model": model, "checkpoint": str(path or ""), **metrics})
    return rows


def format_table(rows: Sequence[dict[str, Any]], num_modes: int) -> str:
    df = pd.DataFrame(rows, columns=list(REPORT_COLUMNS)).drop(columns=["checkpoint"])
    df = df.rename(columns={"min_ade": f"minADE@{num_modes}", "min_fde": f"minFDE@{num_modes}", "ade": "ADE", "fde": "FDE"})
    return df.to_string(index=False, float_format=lambda v: f"{v:.3f}")


def write_report(rows: Sequence[dict[str, Any]], output: Path, extra: Optional[dict[str, Any]] = None) -> tuple[Path, Path]:
    """Write ``output.json`` (rows + settings) and ``output.csv`` (rows)."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    csv_path = output.with_suffix(".csv")
    json_path.write_text(json.dumps({"settings": extra or {}, "results": list(rows)}, indent=2, default=_json_default))
    pd.DataFrame(rows, columns=list(REPORT_COLUMNS)).to_csv(csv_path, index=False)
    return json_path, csv_path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_checkpoint_spec(spec: str) -> tuple[str, str]:
    """``"label=path"`` -> ``(label, path)``; a bare path uses its parent dir name + stem as label."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), path.strip()
    path = Path(spec)
    return f"{path.parent.name}/{path.stem}", spec


def _build_dataset(args: argparse.Namespace, fps: float, obs_len: int, pred_len: int) -> BaseTrajectoryDataset:
    common = dict(obs_len=obs_len, pred_len=pred_len, fps=fps, stride=args.stride, augment=False)
    if args.dataset == "ethucy":
        return ETHUCYDataset(root=args.data_root, test_scene=args.test_scene, split=args.split, **common)
    if not args.csv:
        raise SystemExit("--csv is required for --dataset own_csv")
    return OwnCSVDataset(args.csv, source_fps=args.source_fps, **common)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TP checkpoints against the constant-velocity baseline.")
    parser.add_argument("--dataset", choices=("ethucy", "own_csv"), default="own_csv")
    parser.add_argument("--data-root", default=str(DEFAULT_ETHUCY_ROOT), help="ETH/UCY root directory.")
    parser.add_argument("--test-scene", choices=ETHUCY_SCENES, default="eth")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test", help="ETH/UCY split.")
    parser.add_argument("--csv", nargs="+", default=None, help="Own trajectory CSV files and/or directories.")
    parser.add_argument("--source-fps", type=float, default=10.0, help="CSV frame rate if no timestamp column.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="[LABEL=]PATH",
        help="Checkpoint to evaluate (repeatable). The model type is read from the sidecar .json.",
    )
    parser.add_argument("--model", choices=("social_lstm", "social_gan"), default=None, help="Model for checkpoints without sidecar.")
    parser.add_argument("--fps", type=float, default=None, help="Evaluation fps (default: from checkpoint metadata, else 10).")
    parser.add_argument("--obs-len", type=int, default=None, help="Default: from checkpoint metadata, else 20.")
    parser.add_argument("--pred-len", type=int, default=None, help="Default: from checkpoint metadata, else 30.")
    parser.add_argument("--num-modes", type=int, default=20, help="K for minADE@K / minFDE@K.")
    parser.add_argument("--stride", type=int, default=None, help="Window stride in frames (default: fps / 2.5).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Report path stem (writes .json and .csv).")
    return parser


def main(argv: Optional[list[str]] = None) -> list[dict[str, Any]]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    checkpoints: list[tuple[str, str, Optional[str]]] = []
    first_meta: Optional[dict[str, Any]] = None
    for spec in args.checkpoint:
        label, path = parse_checkpoint_spec(spec)
        meta = load_checkpoint_metadata(path)
        model = (meta or {}).get("model") or args.model
        if not model:
            raise SystemExit(f"Cannot determine the model of '{path}' (no sidecar .json); pass --model.")
        first_meta = first_meta or meta
        checkpoints.append((label, model, path))

    fps = args.fps if args.fps is not None else float((first_meta or {}).get("fps", 10.0))
    obs_len = args.obs_len if args.obs_len is not None else int((first_meta or {}).get("obs_len", 20))
    pred_len = args.pred_len if args.pred_len is not None else int((first_meta or {}).get("pred_len", 30))

    for label, model, path in checkpoints:
        for warning in checkpoint_settings_warnings(
            load_checkpoint_metadata(path), obs_len=obs_len, pred_len=pred_len, fps=fps, model_name=model
        ):
            print(f"WARNING [{label}]: {warning}")

    dataset = _build_dataset(args, fps, obs_len, pred_len)
    windows = dataset.windows
    if not windows:
        raise SystemExit("The evaluation dataset produced no windows (tracks shorter than obs_len + pred_len?).")

    rows = run_evaluation(windows, checkpoints, num_modes=args.num_modes, pred_len=pred_len, seed=args.seed)
    print(f"\nDataset: {args.dataset} ({len(dataset.recordings)} recordings, {len(windows)} windows) | "
          f"fps={fps:g} obs_len={obs_len} pred_len={pred_len} K={args.num_modes}")
    print(format_table(rows, args.num_modes))

    tag = f"{args.dataset}_{args.test_scene}_{args.split}" if args.dataset == "ethucy" else "own_csv"
    output = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT / f"eval_{tag}"
    settings = {
        "dataset": args.dataset,
        "test_scene": args.test_scene if args.dataset == "ethucy" else None,
        "split": args.split if args.dataset == "ethucy" else None,
        "recordings": dataset.recordings,
        "fps": fps,
        "obs_len": obs_len,
        "pred_len": pred_len,
        "num_modes": args.num_modes,
    }
    json_path, csv_path = write_report(rows, output, settings)
    print(f"\nReport written to {json_path} and {csv_path}")
    return rows


if __name__ == "__main__":
    main()
