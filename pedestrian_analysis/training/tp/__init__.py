"""Trajectory-prediction (TP) training pipeline.

Trains the repo's :class:`pipeline.tp_models.SocialLSTMNet` and
:class:`pipeline.tp_models.SocialGANNet` so that the resulting checkpoints
load directly into :class:`pipeline.tp_adapters.SocialLSTMAdapter` /
:class:`pipeline.tp_adapters.SocialGANAdapter` (and the TP Analysis tab).

Modules:
    resample        -- fps resampling of trajectory DataFrames
    datasets        -- dataset loaders (ETH/UCY, own CSV exports, PIE stub),
                       sliding-window extraction and augmentation
    losses          -- best-of-K variety loss + adversarial losses
    metrics         -- ADE / FDE / minADE@K / minFDE@K
    discriminator   -- Social-GAN trajectory discriminator (training only)
    train           -- pretraining CLI (``python -m training.tp.train``)
    finetune        -- fine-tuning with per-recording cross-validation
    evaluate        -- evaluation CLI incl. constant-velocity baseline
    download_ethucy -- fetches the ETH/UCY benchmark files

See ``docs/TP_TRAINING.md`` for the end-to-end workflow.
"""
