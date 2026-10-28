"""Download the ETH/UCY trajectory benchmark (SGAN / Social-STGCNN split layout).

Fetches the standard leave-one-scene-out files (``frame ped_id x y``,
2.5 Hz, metres) from the Social-STGCNN repository -- the same files as the
``datasets/`` folder of https://github.com/agrimgupta92/sgan -- into::

    pedestrian_analysis/data/ethucy/<scene>/{train,val,test}/*.txt

for scene in eth, hotel, univ, zara1, zara2. The data folder is git-ignored;
do not commit it. Usage (from ``pedestrian_analysis/``)::

    python -m training.tp.download_ethucy
    python -m training.tp.download_ethucy --output-dir /path/to/ethucy --scenes eth zara1
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "data" / "ethucy"
BASE_URL = "https://raw.githubusercontent.com/abduallahmohamed/Social-STGCNN/master/datasets"

# Recording files per scene (the test split of a scene = its own files; the
# train/val splits = "<file>_train.txt" / "<file>_val.txt" of all other scenes).
SCENE_FILES: dict[str, tuple[str, ...]] = {
    "eth": ("biwi_eth",),
    "hotel": ("biwi_hotel",),
    "univ": ("students001", "students003"),
    "zara1": ("crowds_zara01",),
    "zara2": ("crowds_zara02",),
}
# Extra recordings that are only ever used for training (never a test scene).
TRAIN_ONLY_FILES: tuple[str, ...] = ("crowds_zara03", "uni_examples")


def split_file_list(test_scene: str) -> dict[str, list[str]]:
    """Return ``{"train": [...], "val": [...], "test": [...]}`` file names for a leave-one-out split."""
    others = [f for scene, files in SCENE_FILES.items() if scene != test_scene for f in files]
    others += list(TRAIN_ONLY_FILES)
    return {
        "train": sorted(f"{name}_train.txt" for name in others),
        "val": sorted(f"{name}_val.txt" for name in others),
        "test": [f"{name}.txt" for name in SCENE_FILES[test_scene]],
    }


def _download(url: str, dest: Path, timeout: float) -> None:
    if not url.startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS URL: {url}")
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS URL
        data = response.read()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)


def download_ethucy(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    scenes: Sequence[str] = tuple(SCENE_FILES),
    base_url: str = BASE_URL,
    overwrite: bool = False,
    timeout: float = 60.0,
) -> int:
    """Download the split files; returns the number of files fetched (existing files are skipped)."""
    output_dir = Path(output_dir)
    fetched = 0
    for scene in scenes:
        if scene not in SCENE_FILES:
            raise ValueError(f"Unknown scene '{scene}'. Choose from {tuple(SCENE_FILES)}.")
        for split, names in split_file_list(scene).items():
            split_dir = output_dir / scene / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                dest = split_dir / name
                if dest.is_file() and dest.stat().st_size > 0 and not overwrite:
                    continue
                url = f"{base_url}/{scene}/{split}/{name}"
                print(f"  {url}")
                _download(url, dest, timeout)
                fetched += 1
    return fetched


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Download the ETH/UCY benchmark files (leave-one-scene-out splits).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scenes", nargs="+", default=list(SCENE_FILES), choices=list(SCENE_FILES))
    parser.add_argument("--base-url", default=BASE_URL, help="Mirror with the same <scene>/<split>/<file> layout.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    print(f"Downloading ETH/UCY into {args.output_dir} ...")
    try:
        count = download_ethucy(args.output_dir, args.scenes, args.base_url, args.overwrite)
    except OSError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Done ({count} files downloaded).")


if __name__ == "__main__":
    main()
