#!/usr/bin/env python3
"""Helper script to fetch official TP model repositories/checkpoints for reference.

IMPORTANT: downloaded checkpoints are trained with the *official* model code
(different network architectures/state_dict keys) and will NOT load into
this repo's from-scratch adapters in pipeline.tp_adapters (SocialLSTMAdapter,
SocialGANAdapter). They are useful for inspection/reference or for a future
effort to vendor the official model code. See pipeline.tp_model_registry for
per-model notes on checkpoint compatibility.

Usage:
    python -m scripts.download_tp_models --model social_gan --output-dir models/trajectory_prediction/
    python -m scripts.download_tp_models --model social_stgcnn --output-dir models/trajectory_prediction/
    python -m scripts.download_tp_models --model all --output-dir models/trajectory_prediction/
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def download_social_gan(output_dir: Path) -> None:
    """Clone agrimgupta92/sgan and copy its pretrained checkpoints."""
    print("Fetching Social-GAN repository and pretrained checkpoints...")
    sgan_dir = output_dir / "sgan_temp"
    if sgan_dir.exists():
        shutil.rmtree(sgan_dir)

    subprocess.run(["git", "clone", "https://github.com/agrimgupta92/sgan.git", str(sgan_dir)], check=True)
    subprocess.run(["bash", "scripts/download_models.sh"], cwd=sgan_dir, check=True)

    models_dir = output_dir / "social_gan"
    models_dir.mkdir(parents=True, exist_ok=True)
    sgan_models = sgan_dir / "sgan-models"
    if sgan_models.exists():
        for checkpoint in sgan_models.glob("*.pt"):
            shutil.copy2(checkpoint, models_dir / checkpoint.name)
        print(f"Checkpoints copied to: {models_dir}")
    else:
        print("Warning: sgan-models directory not found after cloning.")

    shutil.rmtree(sgan_dir, ignore_errors=True)
    print("Social-GAN fetch complete.")


def download_social_stgcnn(output_dir: Path) -> None:
    """Clone abduallahmohamed/Social-STGCNN and copy its bundled checkpoints."""
    print("Fetching Social-STGCNN repository and pretrained checkpoints...")
    stgcnn_dir = output_dir / "Social-STGCNN_temp"
    if stgcnn_dir.exists():
        shutil.rmtree(stgcnn_dir)

    subprocess.run(
        ["git", "clone", "https://github.com/abduallahmohamed/Social-STGCNN.git", str(stgcnn_dir)], check=True
    )

    checkpoint_dir = stgcnn_dir / "checkpoint"
    models_dir = output_dir / "social_stgcnn"
    if checkpoint_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        for checkpoint in checkpoint_dir.rglob("*.pth"):
            shutil.copy2(checkpoint, models_dir / checkpoint.name)
        print(f"Checkpoints copied to: {models_dir}")
    else:
        print("Warning: checkpoint directory not found in cloned repository.")

    shutil.rmtree(stgcnn_dir, ignore_errors=True)
    print("Social-STGCNN fetch complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch official TP model repositories/checkpoints for reference.")
    parser.add_argument("--model", required=True, choices=["social_gan", "social_stgcnn", "all"])
    parser.add_argument("--output-dir", type=Path, default=Path("models/trajectory_prediction"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model in ("social_gan", "all"):
        download_social_gan(args.output_dir)
    if args.model in ("social_stgcnn", "all"):
        download_social_stgcnn(args.output_dir)

    print(f"\nDone. Files stored in: {args.output_dir}")
    print(
        "Reminder: these official checkpoints are NOT compatible with this repo's "
        "adapters (pipeline.tp_adapters). See pipeline.tp_model_registry for details."
    )


if __name__ == "__main__":
    main()
