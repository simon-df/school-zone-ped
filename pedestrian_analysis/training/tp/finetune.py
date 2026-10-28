"""Fine-tune a pretrained TP checkpoint on your own recordings with per-recording CV.

With only a handful of recordings (e.g. 12 school-zone crossings), windows
from the same recording are highly correlated, so data is **always split by
recording (file), never by window**:

* ``--cv-folds 0`` (default): leave-one-recording-out (LORO) CV.
* ``--cv-folds k``: k folds of whole recordings.

Within each fold one or more of the remaining training recordings is held
out for early stopping (``--val-fraction``). For every fold the held-out test
recording(s) are evaluated with the constant-velocity baseline, the
pretrained checkpoint (zero-shot) and the fine-tuned checkpoint; the report
shows mean +/- std across folds. Finally (unless ``--no-final-fit``) a model
is fine-tuned on *all* recordings for the median best epoch of the folds and
saved to ``<output-dir>/final/best.pt`` -- that is the checkpoint to load in
the TP Analysis tab.

Heavy augmentation (rotation, flips, scale jitter) is on by default.

Example (from ``pedestrian_analysis/``)::

    python -m training.tp.finetune \\
        --init-checkpoint outputs/tp_training/social_lstm_ethucy_eth/best.pt \\
        --csv data/trajectories/ --epochs 30 --lr 1e-4 --freeze-encoder
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from training.tp.datasets import OwnCSVDataset, split_recordings  # noqa: E402
from training.tp.evaluate import run_evaluation  # noqa: E402
from training.tp.metrics import METRIC_KEYS, mean_std  # noqa: E402
from training.tp.train import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    TrainConfig,
    add_common_arguments,
    config_from_args,
    fit,
    parse_args_with_config,
)

logger = logging.getLogger(__name__)


def make_folds(recordings: Sequence[str], cv_folds: int, seed: int) -> list[list[str]]:
    """Group recordings into test folds. ``cv_folds <= 0`` or ``>= n`` -> leave-one-recording-out."""
    names = sorted(recordings)
    if len(names) < 2:
        raise ValueError(f"Cross-validation needs at least 2 recordings, got {len(names)}.")
    if cv_folds <= 0 or cv_folds >= len(names):
        return [[name] for name in names]
    order = np.random.default_rng(seed).permutation(len(names))
    return [sorted(names[i] for i in chunk) for chunk in np.array_split(order, cv_folds)]


def summarize_folds(fold_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean/std of every metric per method (label) across folds."""
    labels = list(dict.fromkeys(row["label"] for row in fold_rows))
    summary: list[dict[str, Any]] = []
    for label in labels:
        rows = [r for r in fold_rows if r["label"] == label]
        entry: dict[str, Any] = {"label": label, "folds": len(rows)}
        for key in METRIC_KEYS:
            entry[f"{key}_mean"], entry[f"{key}_std"] = mean_std(r[key] for r in rows)
        summary.append(entry)
    return summary


def format_summary(summary: Sequence[dict[str, Any]], num_modes: int) -> str:
    names = {"ade": "ADE", "fde": "FDE", "min_ade": f"minADE@{num_modes}", "min_fde": f"minFDE@{num_modes}"}
    table = pd.DataFrame(
        [
            {"method": s["label"], "folds": s["folds"],
             **{names[k]: f"{s[f'{k}_mean']:.3f} ± {s[f'{k}_std']:.3f}" for k in METRIC_KEYS}}
            for s in summary
        ]
    )
    return table.to_string(index=False)


def cross_validate(cfg: TrainConfig, cv_folds: int, val_fraction: float, final_fit: bool) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir) if cfg.output_dir else DEFAULT_OUTPUT_ROOT / f"{cfg.model}_finetune"
    augment = True if cfg.augment is None else cfg.augment
    full = OwnCSVDataset(
        cfg.csv, source_fps=cfg.source_fps, obs_len=cfg.obs_len, pred_len=cfg.pred_len, fps=cfg.fps,
        stride=cfg.stride, min_agents=cfg.min_agents, augment=augment, seed=cfg.seed,
    )
    recordings = [r for r in full.recordings if full.windows_by_recording[r]]
    skipped = sorted(set(full.recordings) - set(recordings))
    if skipped:
        logger.warning("Recordings without any %d-frame window are ignored: %s", cfg.obs_len + cfg.pred_len, skipped)
    folds = make_folds(recordings, cv_folds, cfg.seed)
    logger.info("%d recordings -> %d folds (%s)", len(recordings), len(folds),
                "leave-one-recording-out" if all(len(f) == 1 for f in folds) else f"{cv_folds}-fold")

    fold_rows: list[dict[str, Any]] = []
    fold_info: list[dict[str, Any]] = []
    for idx, test_recs in enumerate(folds):
        remaining = [r for r in recordings if r not in test_recs]
        train_recs, val_recs = (split_recordings(remaining, val_fraction, cfg.seed + idx)
                                if len(remaining) >= 3 else (remaining, []))
        fold_dir = output_dir / f"fold_{idx:02d}"
        logger.info("Fold %d/%d: test=%s val=%s train=%d recordings", idx + 1, len(folds), test_recs, val_recs, len(train_recs))
        fold_cfg = replace(cfg, output_dir=str(fold_dir), augment=augment, resume=None)
        result = fit(
            fold_cfg,
            full.subset(train_recs, augment=augment),
            full.subset(val_recs, augment=False) if val_recs else None,
            fold_dir,
            extra_metadata={"cv_fold": idx, "train_recordings": train_recs, "val_recordings": val_recs,
                            "test_recordings": test_recs},
        )
        checkpoints = []
        if cfg.init_checkpoint:
            checkpoints.append(("zero_shot", cfg.model, cfg.init_checkpoint))
        checkpoints.append(("finetuned", cfg.model, result["best_checkpoint"]))
        rows = run_evaluation(full.subset(test_recs, augment=False).windows, checkpoints,
                              num_modes=cfg.num_modes, pred_len=cfg.pred_len, seed=cfg.seed)
        for row in rows:
            row.update({"fold": idx, "test_recordings": ",".join(test_recs)})
            logger.info("  fold %d %-17s ADE %.3f FDE %.3f minADE %.3f minFDE %.3f", idx, row["label"],
                        row["ade"], row["fde"], row["min_ade"], row["min_fde"])
        fold_rows.extend(rows)
        fold_info.append({"fold": idx, "test": test_recs, "val": val_recs, "train": train_recs,
                          "best_epoch": result["best_epoch"], "best_checkpoint": result["best_checkpoint"]})

    summary = summarize_folds(fold_rows)
    print(f"\nCross-validation on {len(recordings)} recordings ({len(folds)} folds), "
          f"fps={cfg.fps:g} obs_len={cfg.obs_len} pred_len={cfg.pred_len} K={cfg.num_modes}:")
    print(format_summary(summary, cfg.num_modes))

    final_checkpoint: Optional[str] = None
    if final_fit:
        best_epochs = [f["best_epoch"] for f in fold_info if f["best_epoch"] > 0]
        epochs = max(1, int(np.median(best_epochs))) if best_epochs else cfg.epochs
        logger.info("Final fit on all %d recordings for %d epochs (median best epoch of the folds).", len(recordings), epochs)
        final_dir = output_dir / "final"
        final = fit(replace(cfg, epochs=epochs, output_dir=str(final_dir), augment=augment, resume=None),
                    full.subset(recordings, augment=augment), None, final_dir,
                    extra_metadata={"train_recordings": recordings, "cv_summary": summary})
        final_checkpoint = final["best_checkpoint"]
        print(f"\nFinal fine-tuned checkpoint (load this in the TP Analysis tab): {final_checkpoint}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"config": asdict(cfg), "folds": fold_info, "fold_results": fold_rows, "summary": summary,
              "final_checkpoint": final_checkpoint}
    (output_dir / "cv_results.json").write_text(json.dumps(report, indent=2, default=str))
    pd.DataFrame(fold_rows).to_csv(output_dir / "cv_fold_results.csv", index=False)
    pd.DataFrame(summary).to_csv(output_dir / "cv_summary.csv", index=False)
    print(f"CV report: {output_dir / 'cv_results.json'}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a TP checkpoint on own CSVs with per-recording cross-validation.")
    add_common_arguments(parser)
    parser.add_argument("--csv", nargs="+", required=False, default=None, help="Own trajectory CSV files and/or directories.")
    parser.add_argument("--cv-folds", type=int, default=0, help="0 = leave-one-recording-out (default), k = k folds.")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Fraction of each fold's training recordings held out for early stopping.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                        help="Random rotation/flip/scale augmentation (default: on).")
    parser.add_argument("--final-fit", action=argparse.BooleanOptionalAction, default=True,
                        help="Fine-tune a final model on all recordings after CV (default: on).")
    parser.set_defaults(batch_size=8)
    return parser


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args_with_config(build_parser(), argv)
    if not args.csv:
        raise SystemExit("--csv is required (own trajectory CSV files and/or directories).")
    cfg = replace(config_from_args(args), dataset="own_csv")
    return cross_validate(cfg, cv_folds=args.cv_folds, val_fraction=args.val_fraction, final_fit=args.final_fit)


if __name__ == "__main__":
    main()
