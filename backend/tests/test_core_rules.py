from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.core.adaptation import MIN_DWELL_SECONDS, STABILITY_SECONDS, PolicyRuntimeState, evaluate_policy
from app.core.state import HMIStateStore, SequenceRejectedError
from app.protocol.models import (
    AttentionState,
    DriverStateEnvelope,
    DriverStatePayload,
    ExperimentCondition,
    GazeRegion,
    InformationDensity,
    InteractionEventPayload,
    PreferenceProfile,
    SceneSnapshotEnvelope,
    SceneSnapshotPayload,
    UserPreferencesPayload,
)


def base_preferences(profile: PreferenceProfile = PreferenceProfile.SAFETY) -> UserPreferencesPayload:
    return UserPreferencesPayload(profile=profile, explanation_detail_level="medium", show_prediction_preference=True)


def driver(workload: float | None, available: bool = True) -> DriverStatePayload:
    return DriverStatePayload(
        gaze_region=GazeRegion.ROAD if available else GazeRegion.UNKNOWN,
        attention=AttentionState.ATTENTIVE if available else AttentionState.UNKNOWN,
        workload=workload if available else None,
        confidence=0.9 if available else 0.0,
        available=available,
        measurement_time_s=0.0,
    )


def test_invalid_values_rejected() -> None:
    with pytest.raises(ValidationError):
        DriverStatePayload(
            gaze_region=GazeRegion.ROAD,
            attention=AttentionState.ATTENTIVE,
            workload=1.4,
            confidence=0.8,
            available=True,
            measurement_time_s=0.0,
        )


def test_static_condition_ignores_workload() -> None:
    runtime = PolicyRuntimeState()
    result = evaluate_policy(
        condition=ExperimentCondition.STATIC,
        driver_state=driver(0.95),
        user_preferences=base_preferences(),
        previous_policy_state=runtime,
        logical_time_s=10.0,
        has_critical_event=False,
        manual_detail_requested=False,
        scene_stale=False,
        input_sequences={},
    )
    assert result.policy.information_density == InformationDensity.MEDIUM
    assert "STATIC_BASELINE" in result.policy.reason_codes


def test_high_workload_stability_and_dwell() -> None:
    runtime = PolicyRuntimeState()
    initial = evaluate_policy(
        condition=ExperimentCondition.ADAPTIVE,
        driver_state=driver(0.8),
        user_preferences=base_preferences(),
        previous_policy_state=runtime,
        logical_time_s=0.5,
        has_critical_event=False,
        manual_detail_requested=False,
        scene_stale=False,
        input_sequences={},
    )
    assert initial.policy.information_density == InformationDensity.MEDIUM

    stabilized = evaluate_policy(
        condition=ExperimentCondition.ADAPTIVE,
        driver_state=driver(0.82),
        user_preferences=base_preferences(),
        previous_policy_state=initial.runtime_state,
        logical_time_s=STABILITY_SECONDS + MIN_DWELL_SECONDS + 0.1,
        has_critical_event=False,
        manual_detail_requested=False,
        scene_stale=False,
        input_sequences={},
    )
    assert stabilized.policy.information_density == InformationDensity.LOW
    assert "HIGH_WORKLOAD" in stabilized.policy.reason_codes


def test_hysteresis_prevents_flapping() -> None:
    runtime = PolicyRuntimeState()
    initial = evaluate_policy(
        condition=ExperimentCondition.ADAPTIVE,
        driver_state=driver(0.9),
        user_preferences=base_preferences(),
        previous_policy_state=runtime,
        logical_time_s=1.0,
        has_critical_event=False,
        manual_detail_requested=False,
        scene_stale=False,
        input_sequences={},
    )
    low_density = evaluate_policy(
        condition=ExperimentCondition.ADAPTIVE,
        driver_state=driver(0.9),
        user_preferences=base_preferences(),
        previous_policy_state=initial.runtime_state,
        logical_time_s=5.5,
        has_critical_event=False,
        manual_detail_requested=False,
        scene_stale=False,
        input_sequences={},
    )
    assert low_density.policy.information_density == InformationDensity.LOW
    medium_attempt = evaluate_policy(
        condition=ExperimentCondition.ADAPTIVE,
        driver_state=driver(0.65),
        user_preferences=base_preferences(),
        previous_policy_state=low_density.runtime_state,
        logical_time_s=5.9,
        has_critical_event=False,
        manual_detail_requested=False,
        scene_stale=False,
        input_sequences={},
    )
    assert medium_attempt.policy.information_density == InformationDensity.LOW


def test_minimum_dwell_blocks_immediate_return() -> None:
    runtime = PolicyRuntimeState(density=InformationDensity.LOW, candidate_density=InformationDensity.LOW, candidate_since_s=0.0, last_density_change_s=1.0)
    result = evaluate_policy(
        condition=ExperimentCondition.ADAPTIVE,
        driver_state=driver(0.2),
        user_preferences=base_preferences(),
        previous_policy_state=runtime,
        logical_time_s=2.0,
        has_critical_event=False,
        manual_detail_requested=False,
        scene_stale=False,
        input_sequences={},
    )
    assert result.policy.information_density == InformationDensity.LOW


def test_critical_event_visible_at_low_density() -> None:
    result = evaluate_policy(
        condition=ExperimentCondition.ADAPTIVE,
        driver_state=driver(0.95),
        user_preferences=base_preferences(),
        previous_policy_state=PolicyRuntimeState(density=InformationDensity.LOW, candidate_density=InformationDensity.LOW, candidate_since_s=0.0, last_density_change_s=0.0),
        logical_time_s=10.0,
        has_critical_event=True,
        manual_detail_requested=False,
        scene_stale=False,
        input_sequences={},
    )
    assert "CRITICAL_EVENT" in result.policy.reason_codes
    assert result.policy.information_density == InformationDensity.LOW


def test_missing_dms_treated_as_unknown() -> None:
    result = evaluate_policy(
        condition=ExperimentCondition.ADAPTIVE,
        driver_state=driver(None, available=False),
        user_preferences=base_preferences(),
        previous_policy_state=PolicyRuntimeState(),
        logical_time_s=5.0,
        has_critical_event=False,
        manual_detail_requested=False,
        scene_stale=False,
        input_sequences={},
    )
    assert "DRIVER_STATE_UNKNOWN" in result.policy.reason_codes
    assert result.policy.information_density == InformationDensity.MEDIUM


@pytest.mark.asyncio
async def test_duplicate_sequence_rejected() -> None:
    store = HMIStateStore()
    envelope = DriverStateEnvelope(
        schema_version="0.1",
        type="driver.state",
        session_id="a",
        source_id="source",
        sequence=1,
        simulation_time_s=1.0,
        payload=driver(0.4),
    )
    await store.ingest(envelope)
    with pytest.raises(SequenceRejectedError):
        await store.ingest(envelope)


def _scene(track_ids: list[str]) -> SceneSnapshotPayload:
    from app.protocol.models import EgoPlan, EgoState, LanePolyline, ObjectClass, ObjectTrack, Point3D, Provenance

    tracks = [
        ObjectTrack(
            track_id=track_id,
            object_class=ObjectClass.PEDESTRIAN,
            position=Point3D(x_m=1.0, y_m=1.0, z_m=0.0),
            yaw_rad=0.0,
            extent_m=Point3D(x_m=0.4, y_m=0.4, z_m=1.7),
            velocity_mps=Point3D(x_m=0.0, y_m=0.0, z_m=0.0),
        )
        for track_id in track_ids
    ]
    return SceneSnapshotPayload(
        ego=EgoState(speed_mps=3.0, pose=Point3D(x_m=0.0, y_m=0.0, z_m=0.0), yaw_rad=0.0),
        tracks=tracks,
        lanes=[LanePolyline(lane_id="a", points=[Point3D(x_m=0.0, y_m=0.0, z_m=0.0)])],
        ego_plan=EgoPlan(maneuver="keep_lane", trajectory=[Point3D(x_m=1.0, y_m=0.0, z_m=0.0)]),
        provenance=Provenance.SYNTHETIC,
    )


@pytest.mark.asyncio
async def test_full_snapshot_removes_disappeared_objects() -> None:
    store = HMIStateStore()
    await store.apply_interaction(
        InteractionEventPayload(
            interaction_id="i1",
            display_id="center",
            action="select_object",
            value="track-a",
        )
    )
    first = SceneSnapshotEnvelope(
        schema_version="0.1",
        type="scene.snapshot",
        session_id="s1",
        source_id="mock",
        sequence=1,
        simulation_time_s=1.0,
        payload=_scene(["track-a", "track-b"]),
    )
    await store.ingest(first)
    state_first = await store.snapshot()
    assert state_first.scene is not None
    assert {track.track_id for track in state_first.scene.tracks} == {"track-a", "track-b"}

    second = SceneSnapshotEnvelope(
        schema_version="0.1",
        type="scene.snapshot",
        session_id="s1",
        source_id="mock",
        sequence=2,
        simulation_time_s=2.0,
        payload=_scene(["track-b"]),
    )
    await store.ingest(second)
    state_second = await store.snapshot()
    assert state_second.scene is not None
    assert {track.track_id for track in state_second.scene.tracks} == {"track-b"}
    assert state_second.selected_track_id is None


@pytest.mark.asyncio
async def test_stale_scene_marked() -> None:
    store = HMIStateStore()
    await store.ingest(
        SceneSnapshotEnvelope(
            schema_version="0.1",
            type="scene.snapshot",
            session_id="s1",
            source_id="mock",
            sequence=1,
            simulation_time_s=0.0,
            payload=_scene(["track-a"]),
        )
    )
    await store.ingest(
        DriverStateEnvelope(
            schema_version="0.1",
            type="driver.state",
            session_id="s1",
            source_id="mock",
            sequence=1,
            simulation_time_s=2.0,
            payload=driver(0.4),
        )
    )
    state = await store.snapshot()
    assert state.system_status.scene_data_stale is True
    assert state.policy is not None
    assert "SCENE_DATA_STALE" in state.policy.reason_codes


def test_decisions_deterministic() -> None:
    runtime = PolicyRuntimeState()
    inputs = dict(
        condition=ExperimentCondition.ADAPTIVE,
        driver_state=driver(0.3),
        user_preferences=base_preferences(PreferenceProfile.TECH),
        previous_policy_state=runtime,
        logical_time_s=8.0,
        has_critical_event=False,
        manual_detail_requested=True,
        scene_stale=False,
        input_sequences={"mock::driver.state": 4},
    )
    one = evaluate_policy(**deepcopy(inputs))
    two = evaluate_policy(**deepcopy(inputs))
    p1 = one.policy.model_dump(exclude={"policy_id"})
    p2 = two.policy.model_dump(exclude={"policy_id"})
    assert p1 == p2
