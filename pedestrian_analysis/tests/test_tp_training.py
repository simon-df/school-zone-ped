"""Tests for the TP training pipeline (training.tp): datasets, resampling, losses, training, checkpoints."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from pipeline.tp_adapters import SocialGANAdapter, SocialLSTMAdapter  # noqa: E402
from pipeline.tp_model_registry import (  # noqa: E402
    TPModelRegistry,
    checkpoint_settings_warnings,
    load_checkpoint_metadata,
)
from pipeline.tp_models import SocialGANNet, SocialLSTMNet  # noqa: E402
from training.tp.datasets import (  # noqa: E402
    AugmentationConfig,
    ETHUCYDataset,
    OwnCSVDataset,
    PIEDataset,
    augment_window,
    extract_windows,
    load_ethucy_file,
    normalize_window_translation,
    split_recordings,
)
from training.tp.evaluate import run_evaluation  # noqa: E402
from training.tp.finetune import make_folds, summarize_folds  # noqa: E402
from training.tp.losses import discriminator_loss, generator_adversarial_loss, variety_loss  # noqa: E402
from training.tp.metrics import displacement_errors  # noqa: E402
from training.tp.resample import resample_trajectories  # noqa: E402
from training.tp.train import TrainConfig, fit, target_diffs  # noqa: E402

_TINY = dict(embedding_dim=8, hidden_dim=8)


def _linear_tracks(num_tracks: int = 3, num_frames: int = 30, fps: float = 10.0, offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for tid in range(num_tracks):
        for f in range(num_frames):
            rows.append({"id": tid, "frame": f, "timestamp": f / fps, "x": offset + 0.1 * f + tid, "y": 0.05 * f - tid})
    return pd.DataFrame(rows)


def _write_recordings(tmp_path, count: int = 3) -> list[str]:
    paths = []
    for i in range(count):
        path = tmp_path / f"rec_{i}.csv"
        df = _linear_tracks(num_tracks=2, num_frames=20, offset=float(i))
        df.rename(columns={"id": "track_id", "x": "x_m", "y": "y_m"}).to_csv(path, index=False)
        paths.append(str(path))
    return paths


# ----------------------------------------------------------------------
# resampling + windowing
# ----------------------------------------------------------------------


def test_resample_upsampling_keeps_linear_trajectory_linear() -> None:
    df = _linear_tracks(num_tracks=1, num_frames=5, fps=2.5)  # 0 .. 1.6 s
    out = resample_trajectories(df, target_fps=10.0)

    assert list(out.columns) == ["id", "frame", "timestamp", "x", "y"]
    assert out["frame"].tolist() == list(range(17))
    np.testing.assert_allclose(np.diff(out["x"].to_numpy()), 0.1 * 2.5 / 10.0)
    np.testing.assert_allclose(out["x"].to_numpy(), 0.1 * 2.5 * out["timestamp"].to_numpy())


def test_resample_decimation_picks_original_samples() -> None:
    df = _linear_tracks(num_tracks=1, num_frames=21, fps=10.0)
    out = resample_trajectories(df, target_fps=2.5)

    assert out["frame"].tolist() == [0, 1, 2, 3, 4, 5]
    np.testing.assert_allclose(out["x"].to_numpy(), df["x"].to_numpy()[::4])


def test_resample_does_not_interpolate_across_large_gaps() -> None:
    df = _linear_tracks(num_tracks=1, num_frames=20)
    df = df[(df["frame"] < 5) | (df["frame"] >= 15)]
    out = resample_trajectories(df, target_fps=10.0)

    assert set(out["frame"]) == set(range(5)) | set(range(15, 20))


def test_extract_windows_shapes_stride_and_short_tracks() -> None:
    df = _linear_tracks(num_tracks=2, num_frames=12)
    short = pd.DataFrame({"id": 9, "frame": range(3), "x": 0.0, "y": 0.0})
    windows = extract_windows(pd.concat([df, short]), obs_len=4, pred_len=3, stride=2, recording="r")

    assert len(windows) == 3  # starts 0, 2, 4 (seq_len 7 within 12 frames)
    assert windows[0].obs.shape == (2, 4, 2) and windows[0].fut.shape == (2, 3, 2)
    assert 9 not in windows[0].agent_ids  # too short -> skipped
    np.testing.assert_allclose(windows[1].obs[0, 0], df[(df["id"] == 0) & (df["frame"] == 2)][["x", "y"]].to_numpy()[0])
    assert all(w.recording == "r" for w in windows)


def test_augmentation_preserves_relative_geometry() -> None:
    window = extract_windows(_linear_tracks(num_tracks=2, num_frames=8), obs_len=4, pred_len=4)[0]
    norm = normalize_window_translation(window)
    np.testing.assert_allclose(norm.obs[:, -1].mean(axis=0), 0.0, atol=1e-12)

    aug = augment_window(norm, np.random.default_rng(0), AugmentationConfig(scale_range=(1.0, 1.0)))
    dist = lambda w: np.linalg.norm(w.obs[0] - w.obs[1], axis=-1)  # noqa: E731
    np.testing.assert_allclose(dist(aug), dist(window))
    assert not np.allclose(aug.obs, norm.obs)


def test_own_csv_dataset_splits_by_recording(tmp_path) -> None:
    paths = _write_recordings(tmp_path, count=3)
    ds = OwnCSVDataset(paths, obs_len=4, pred_len=4, fps=10.0, stride=4)

    assert ds.recordings == ["rec_0", "rec_1", "rec_2"]
    train, val = split_recordings(ds.recordings, 0.34, seed=0)
    assert len(val) == 1 and not set(train) & set(val)
    sub = ds.subset(val)
    assert {w.recording for w in sub.windows} == set(val)
    assert sub[0].obs.shape == (2, 4, 2)


def test_ethucy_loader_parses_sgan_format(tmp_path) -> None:
    split_dir = tmp_path / "eth" / "train"
    split_dir.mkdir(parents=True)
    lines = [f"{f * 10}\t1.0\t{0.4 * f:.2f}\t1.0" for f in range(25)]
    (split_dir / "biwi_hotel_train.txt").write_text("\n".join(lines) + "\n")

    df = load_ethucy_file(split_dir / "biwi_hotel_train.txt")
    np.testing.assert_allclose(df["timestamp"].to_numpy()[:3], [0.0, 0.4, 0.8])

    ds = ETHUCYDataset(tmp_path, "eth", "train", obs_len=8, pred_len=12, fps=2.5)
    assert len(ds) == 6 and ds[0].obs.shape == (1, 8, 2)
    ds10 = ETHUCYDataset(tmp_path, "eth", "train", obs_len=20, pred_len=30, fps=10.0, stride=1)
    assert ds10[0].fut.shape == (1, 30, 2)

    with pytest.raises(FileNotFoundError, match="download_ethucy"):
        _ = ETHUCYDataset(tmp_path, "zara1", "train").windows


def test_pie_dataset_is_a_stub() -> None:
    with pytest.raises(NotImplementedError, match="ego-vehicle"):
        PIEDataset()


# ----------------------------------------------------------------------
# losses + metrics
# ----------------------------------------------------------------------


def test_variety_loss_picks_best_mode() -> None:
    gt = torch.ones(2, 3, 2)
    pred = torch.zeros(2, 4, 3, 2)
    pred[:, 2] = 1.0  # mode 2 is perfect
    assert float(variety_loss(pred, gt)) == pytest.approx(0.0)
    assert float(variety_loss(pred[:, :1], gt)) > 0.0

    pred.requires_grad_(True)
    variety_loss(pred + 0.1, gt).backward()
    assert pred.grad[:, 2].abs().sum() > 0 and pred.grad[:, 0].abs().sum() == 0


def test_gan_losses() -> None:
    confident = torch.tensor([10.0, 10.0])
    assert float(generator_adversarial_loss(confident)) < 1e-3
    assert float(discriminator_loss(confident, -confident)) < 1e-3
    assert float(discriminator_loss(-confident, confident)) > 10.0


def test_target_diffs_start_at_last_observation() -> None:
    obs = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    fut = torch.tensor([[[3.0, 0.0], [6.0, 0.0]]])
    np.testing.assert_allclose(target_diffs(obs, fut).numpy(), [[[2.0, 0.0], [3.0, 0.0]]])


def test_displacement_errors() -> None:
    gt = np.zeros((1, 2, 2))
    pred = np.stack([np.full((2, 2), 3.0), np.zeros((2, 2))])[None]  # mode0 off, mode1 perfect
    errs = displacement_errors(pred, gt)
    assert errs["ade"][0] == pytest.approx(np.sqrt(18)) and errs["min_ade"][0] == 0.0


# ----------------------------------------------------------------------
# social pooling
# ----------------------------------------------------------------------


@pytest.mark.parametrize("net_cls", [SocialLSTMNet, SocialGANNet])
def test_social_pooling_is_optional_and_interaction_aware(net_cls) -> None:
    plain = net_cls(**_TINY)
    pooled = net_cls(**_TINY, use_social_pooling=True)
    assert not any(k.startswith(("social_pool", "pool_proj")) for k in plain.state_dict())
    pooled.load_state_dict(plain.state_dict(), strict=False)
    torch.nn.init.normal_(pooled.pool_proj.weight)

    diffs = torch.randn(3, 4, 2)
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
    h_together, _ = pooled.encode(diffs, last_pos=pos)
    h_apart, _ = pooled.encode(diffs, last_pos=pos, scene_ids=torch.tensor([0, 1, 2]))
    assert not torch.allclose(h_together, h_apart)
    with pytest.raises(ValueError, match="last_pos"):
        pooled.encode(diffs)


# ----------------------------------------------------------------------
# smoke training + checkpoint compatibility with the adapters
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,adapter_cls,pooling",
    [
        ("social_lstm", SocialLSTMAdapter, False),
        ("social_gan", SocialGANAdapter, False),
        ("social_lstm", SocialLSTMAdapter, True),
        ("social_gan", SocialGANAdapter, True),
    ],
)
def test_smoke_train_checkpoint_loads_into_adapter(tmp_path, model, adapter_cls, pooling) -> None:
    paths = _write_recordings(tmp_path, count=3)
    full = OwnCSVDataset(paths, obs_len=4, pred_len=3, fps=10.0, stride=2, augment=True)
    cfg = TrainConfig(model=model, dataset="own_csv", obs_len=4, pred_len=3, num_modes=3, max_modes=4,
                      noise_dim=4, epochs=2, batch_size=4, device="cpu", use_social_pooling=pooling, **_TINY)
    out = tmp_path / "run"
    result = fit(cfg, full.subset(["rec_0", "rec_1"]), full.subset(["rec_2"], augment=False), out)

    for name in ("best.pt", "last.pt", "best.json", "last.json", "train_state.pt"):
        assert (out / name).is_file()
    meta = json.loads((out / "best.json").read_text())
    assert meta["model"] == model and meta["fps"] == 10.0 and meta["obs_len"] == 4 and meta["pred_len"] == 3
    assert meta["model_kwargs"]["use_social_pooling"] is pooling
    assert len(result["history"]) == 2

    # Direct adapter construction with the matching architecture kwargs.
    adapter = adapter_cls(checkpoint=result["best_checkpoint"], pred_len=3, **meta["model_kwargs"])
    assert adapter.checkpoint_loaded
    obs = full.windows[0].obs
    assert adapter.predict(obs, num_modes=3, pred_len=3).shape == (obs.shape[0], 3, 3, 2)

    # Registry path (as used by the TP Analysis tab) picks the kwargs up from the sidecar.
    via_registry = TPModelRegistry.create_adapter(model, checkpoint_path=result["best_checkpoint"], pred_len=3)
    assert via_registry.predict(obs, num_modes=2, pred_len=5).shape == (obs.shape[0], 2, 5, 2)

    # Resume continues from the saved training state.
    resumed = fit(TrainConfig(**{**cfg.__dict__, "epochs": 3, "resume": str(out / "train_state.pt")}),
                  full.subset(["rec_0", "rec_1"]), full.subset(["rec_2"], augment=False), out)
    assert [h["epoch"] for h in resumed["history"]] == [1, 2, 3]


def test_default_architecture_checkpoint_loads_into_default_adapters(tmp_path) -> None:
    paths = _write_recordings(tmp_path, count=2)
    ds = OwnCSVDataset(paths, obs_len=4, pred_len=3, fps=10.0, stride=4)
    for model, adapter_cls in (("social_lstm", SocialLSTMAdapter), ("social_gan", SocialGANAdapter)):
        cfg = TrainConfig(model=model, dataset="own_csv", obs_len=4, pred_len=3, num_modes=2, epochs=1, device="cpu")
        result = fit(cfg, ds, None, tmp_path / model)
        adapter = adapter_cls(checkpoint=result["best_checkpoint"])  # stock adapter defaults
        assert adapter.predict(ds.windows[0].obs, num_modes=2, pred_len=3).shape == (2, 2, 3, 2)


def test_init_checkpoint_into_pooling_model_and_freeze_encoder(tmp_path) -> None:
    paths = _write_recordings(tmp_path, count=2)
    ds = OwnCSVDataset(paths, obs_len=4, pred_len=3, fps=10.0, stride=4)
    base = fit(TrainConfig(model="social_lstm", dataset="own_csv", obs_len=4, pred_len=3, num_modes=2, epochs=1,
                           device="cpu", **_TINY), ds, None, tmp_path / "base")
    base_state = torch.load(base["best_checkpoint"])
    tuned = fit(TrainConfig(model="social_lstm", dataset="own_csv", obs_len=4, pred_len=3, num_modes=2, epochs=1,
                            device="cpu", use_social_pooling=True, init_checkpoint=base["best_checkpoint"],
                            freeze_encoder=True, lr=1e-2, **_TINY), ds, None, tmp_path / "tuned")
    tuned_state = torch.load(tuned["best_checkpoint"])
    torch.testing.assert_close(tuned_state["encoder.weight_ih_l0"], base_state["encoder.weight_ih_l0"])
    assert not torch.equal(tuned_state["output_head.weight"], base_state["output_head.weight"])
    assert "social_pool.rel_embed.weight" in tuned_state


def test_checkpoint_metadata_warnings(tmp_path) -> None:
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"")
    (tmp_path / "best.json").write_text(json.dumps({"model": "social_lstm", "fps": 2.5, "obs_len": 8, "pred_len": 12}))

    meta = load_checkpoint_metadata(ckpt)
    assert meta["obs_len"] == 8
    warnings = checkpoint_settings_warnings(meta, obs_len=20, pred_len=12, fps=10.0, model_name="social_lstm")
    assert len(warnings) == 2 and any("2.5 fps" in w for w in warnings)
    assert checkpoint_settings_warnings(meta, obs_len=8, pred_len=12, fps=2.5) == []
    assert load_checkpoint_metadata(tmp_path / "missing.pt") is None


def test_evaluation_includes_constant_velocity_and_cv_helpers(tmp_path) -> None:
    paths = _write_recordings(tmp_path, count=2)
    ds = OwnCSVDataset(paths, obs_len=4, pred_len=3, fps=10.0, stride=4)
    rows = run_evaluation(ds.windows, [], num_modes=2, pred_len=3)
    assert rows[0]["label"] == "constant_velocity"
    assert rows[0]["ade"] == pytest.approx(0.0, abs=1e-9)  # synthetic tracks are perfectly linear

    assert make_folds([f"r{i}" for i in range(12)], 0, seed=0) == [[f"r{i}"] for i in sorted(range(12), key=str)]
    folds = make_folds([f"r{i}" for i in range(12)], 4, seed=0)
    assert len(folds) == 4 and sorted(sum(folds, [])) == sorted(f"r{i}" for i in range(12))
    summary = summarize_folds([{"label": "a", "ade": 1.0, "fde": 2.0, "min_ade": 1.0, "min_fde": 2.0},
                               {"label": "a", "ade": 3.0, "fde": 2.0, "min_ade": 1.0, "min_fde": 2.0}])
    assert summary[0]["ade_mean"] == 2.0 and summary[0]["ade_std"] == 1.0
