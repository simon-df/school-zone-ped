# Trajectory-Prediction Training (Social-LSTM / Social-GAN)

This guide covers training the built-in `social_lstm` and `social_gan` TP models: pretraining on ETH/UCY, zero-shot evaluation on your own drone CSVs, fine-tuning on your own recordings, and loading the result in the **TP Analyse** tab.

All commands run from the `pedestrian_analysis/` directory, which is the import root (`pipeline.`, `training.`, ...):

```bash
cd pedestrian_analysis
pip install -r requirements.txt   # torch, pyyaml, pandas, ...
```

The code lives in `pedestrian_analysis/training/tp/`:

| Module | Purpose |
|---|---|
| `download_ethucy.py` | fetches ETH/UCY (leave-one-scene-out splits) into `data/ethucy/` |
| `resample.py` | fps resampling (linear interpolation / decimation) of trajectory DataFrames |
| `datasets.py` | `BaseTrajectoryDataset` interface, `ETHUCYDataset`, `OwnCSVDataset`, `PIEDataset` stub, sliding windows, augmentation |
| `losses.py` / `metrics.py` | best-of-K variety loss, GAN losses / ADE, FDE, minADE@K, minFDE@K |
| `discriminator.py` | Social-GAN `TrajectoryDiscriminator`, used only during training |
| `train.py` | pretraining CLI |
| `finetune.py` | fine-tuning with leave-one-recording-out / k-fold CV |
| `evaluate.py` | evaluation CLI, always including the constant-velocity baseline |

## 0. Frame rate, `obs_len` and `pred_len` must match

The adapters work on frame-to-frame displacements, so a checkpoint is tied to the **frame rate** it was trained at:

* ETH/UCY is annotated at **2.5 Hz**. Its standard protocol is obs 8 / pred 12, which is 3.2 s / 4.8 s.
* Your drone CSVs and the TP Analysis tab defaults use **10 Hz** with obs 20 / pred 30, which is 2 s / 3 s.

By default the pipeline resamples ETH/UCY up to 10 Hz by linear interpolation and trains with `--fps 10 --obs-len 20 --pred-len 30`, so checkpoints match the UI defaults. You can also train natively at 2.5 Hz with `--fps 2.5 --obs-len 8 --pred-len 12`. Your own CSVs are resampled to `--fps` using their `timestamp` column.

Every checkpoint gets a sidecar `best.json` / `last.json` file that records the model, architecture kwargs (incl. `use_social_pooling`), `fps`, `obs_len` and `pred_len`. The model registry reads it automatically:

* it builds the adapter with the matching architecture;
* the TP Analysis tab shows a warning when *Obs window*, *Pred frames* or the tab's 10 Hz assumption differ from the training settings.

The tab treats consecutive `frame` values as 10 Hz. If your exported CSV uses the video frame rate (e.g. 25 fps), resample it first:

```bash
python -m training.tp.resample data/trajectories/rec01.csv data/trajectories_10hz/rec01.csv --fps 10
```

## 1. Download ETH/UCY

```bash
python -m training.tp.download_ethucy            # -> pedestrian_analysis/data/ethucy/<scene>/{train,val,test}/
```

This fetches the standard SGAN / Social-STGCNN files (`frame ped_id x y`, metres) for eth, hotel, univ, zara1 and zara2. The folder is git-ignored; never commit it.

## 2. Pretrain (leave-one-scene-out)

`--test-scene` is held out. Training uses the other four scenes, validation uses their `val` split (early stopping), and at the end `best.pt` is evaluated on the held-out scene against constant velocity.

```bash
# Social-LSTM (best-of-K over the K mode embeddings)
python -m training.tp.train --model social_lstm --dataset ethucy --test-scene zara1 \
    --epochs 50 --num-modes 20 --output-dir outputs/tp_training/social_lstm_zara1

# Social-GAN (generator: best-of-K L2 + adversarial loss from an LSTM discriminator)
python -m training.tp.train --model social_gan --dataset ethucy --test-scene zara1 \
    --epochs 100 --num-modes 20 --adv-weight 0.1 --output-dir outputs/tp_training/social_gan_zara1

# With interaction modelling (recommended for groups crossing together)
python -m training.tp.train --model social_gan --dataset ethucy --test-scene zara1 --use-social-pooling
```

Useful options:

* `--fps`, `--obs-len`, `--pred-len`, `--stride` (default `fps / 2.5`)
* `--batch-size` (windows per batch), `--lr`, `--epochs`, `--patience`, `--early-stop-metric {min_ade,ade,...}`
* `--device auto|cpu|cuda`, `--seed`
* `--augment`, `--resume <output-dir>/train_state.pt`
* `--config run.yaml`: a YAML mapping of option names to values. CLI flags override it.

Outputs in `--output-dir`:

* `best.pt` / `last.pt`: plain `state_dict` of the inference net. For Social-GAN this is the **generator only**.
* `best.json` / `last.json`: sidecar metadata.
* `train_state.pt`: full state for `--resume`, including discriminator and optimizers.
* `history.json`, plus `test_results.json` / `.csv` for ETH/UCY runs.

The best epoch is selected on validation **minADE@K** by default, because training optimises best-of-K. "ADE" in the reports is the first mode only. For Social-GAN that is one random sample, so compare models on minADE@K / minFDE@K.

Rough CPU cost at 10 Hz is about 1 minute per epoch for Social-LSTM. A GPU (`--device cuda`) is much faster.

## 3. Zero-shot evaluation on your own CSVs

```bash
python -m training.tp.evaluate --dataset own_csv --csv data/trajectories/ \
    --checkpoint lstm=outputs/tp_training/social_lstm_zara1/best.pt \
    --checkpoint gan=outputs/tp_training/social_gan_zara1/best.pt \
    --num-modes 20 --output outputs/tp_training/eval_own_zero_shot
```

This prints a table and writes `.json` / `.csv` reports. Columns:

* **ADE / FDE**: first mode.
* **minADE@K / minFDE@K**: best of K modes.

The constant-velocity baseline (`DummyTPAdapter`) is always included. Checkpoints are loaded through the same registry path the UI uses.

Accepted CSV columns are `id, frame, timestamp, x, y` or `track_id, frame, timestamp, x_m, y_m` (metric, top-down). Each file is one recording.

## 4. Fine-tune on your own recordings (cross-validated)

```bash
python -m training.tp.finetune \
    --init-checkpoint outputs/tp_training/social_lstm_zara1/best.pt \
    --csv data/trajectories/ --epochs 30 --lr 1e-4 --freeze-encoder \
    --output-dir outputs/tp_training/social_lstm_finetune
```

* **Splits are always by recording (file), never by window.** Windows from the same recording are strongly correlated.
* `--cv-folds 0` is the default and runs leave-one-recording-out (12 folds for 12 recordings). Use `--cv-folds 4` for 4 folds of 3 recordings.
* In each fold, part of the training recordings (`--val-fraction`) is held out for early stopping.
* Every fold reports constant velocity, **zero-shot** (pretrained) and **fine-tuned** results on the held-out recording(s). The summary shows mean ± std across folds; see `cv_results.json`, `cv_summary.csv` and `cv_fold_results.csv`.
* Heavy augmentation (random rotation, flips, 0.9–1.1 scale jitter) is on by default. Disable it with `--no-augment`.
* `--freeze-encoder` freezes the input embedding and the encoder LSTM.
* Architecture, fps, `obs_len` and `pred_len` are inherited from the init checkpoint's sidecar.
* `--use-social-pooling` on top of a non-pooling checkpoint adds a freshly initialised pooling module. Its residual projection starts at zero, so the model initially behaves exactly like the pretrained one.
* After CV, a final model is trained on **all** recordings for the median best epoch of the folds, saved to `<output-dir>/final/best.pt`. Use `--no-final-fit` to skip this step.

**12 recordings is a very small dataset.** Fold-to-fold variance will be large. Judge results by comparing against the constant-velocity baseline on the same folds. If fine-tuning does not beat it consistently, prefer the simpler model.

## 5. Load the checkpoint in the TP Analysis tab

1. Start the app and open **TP Analyse**. Load a (10 Hz) trajectory CSV.
2. Select `social_lstm` or `social_gan`, matching the trained model.
3. Click **Browse...** next to *Checkpoint* and pick `best.pt` (e.g. `outputs/tp_training/social_lstm_finetune/final/best.pt`). Keep its `best.json` next to it.
4. Set *Obs window* / *Pred frames* to the training `obs_len` / `pred_len` (20 / 30 by default). A warning lists any mismatch.

With `use_social_pooling` enabled, all pedestrians visible in the current frame are pooled together, as during training.

Programmatic use:

```python
from pipeline.tp_model_registry import TPModelRegistry
adapter = TPModelRegistry.create_adapter("social_lstm", checkpoint_path=".../best.pt", pred_len=30)
pred = adapter.predict(observed, num_modes=5)   # (N, 5, 30, 2)
```

`SocialLSTMAdapter(checkpoint=".../best.pt")` also works directly with the default architecture. Pass `use_social_pooling=True` and so on yourself if you changed it.

## PIE (not supported yet)

`training.tp.datasets.PIEDataset` is a stub that raises `NotImplementedError`. PIE is filmed from an **ego-vehicle camera** and annotates pedestrians as **image-space bounding boxes** in a moving frame. These models are trained on **top-down metric** trajectories. Adding PIE needs ground-plane projection with ego-motion compensation, or an image-space model. To add any new dataset, subclass `BaseTrajectoryDataset` and implement `load_recordings()`, returning one metric `id/timestamp/x/y` DataFrame per recording. Register it in `DATASET_REGISTRY`.
