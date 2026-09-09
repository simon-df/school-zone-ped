# Tracker adapters for research use-cases

## Selected top-3 trackers (bird's-eye/top-down pedestrian tracking)

For this repository, the recommended top-3 tracker backends are:

1. `bot_sort` (supervision BoT-SORT)
2. `ocsort` (boxmot OC-SORT)
3. `byte_track` (supervision ByteTrack)

### Selection criteria

- **Robustness under occlusion:** BoT-SORT is preferred for stronger re-association and fewer identity breaks in partial occlusion.
- **ID stability for trajectory research:** BoT-SORT and OC-SORT are prioritized for consistent ID continuity in crowded and non-linear motion scenes.
- **Integration complexity / dependencies:** ByteTrack and BoT-SORT integrate directly via `supervision`; OC-SORT is supported through optional `boxmot`.
- **Reproducibility in pipelines:** all three are open, documented trackers with deterministic config-driven usage and clear runtime dependency checks.

## Research tracker groups

The factory supports group names and picks the first tracker that initializes successfully:

- `research_top_down_general`: `bot_sort` -> `ocsort` -> `byte_track`
- `research_top_down_occlusion`: `bot_sort` -> `ocsort` -> `byte_track`
- `research_top_down_small_targets`: `bot_sort` -> `byte_track` -> `ocsort`

If a preferred tracker is unavailable (missing dependency, incompatible install), the factory logs the reason and falls back to the next tracker.

## Usage examples

```python
from pipeline.tracker_adapters import create_tracker_adapter

tracker = create_tracker_adapter("bot_sort")
tracker = create_tracker_adapter("research_top_down_occlusion")
```

## Optional dependency for OC-SORT

```bash
pip install boxmot
```
