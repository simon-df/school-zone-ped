from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.protocol.models import DriverStateEnvelope, SourceMode
from app.recording.manager import RecordingManager
from app.sources.replay_source import ReplaySourceEngine
from app.core.state import HMIStateStore
from pydantic import TypeAdapter


def test_valid_message_accepted_and_schema_rejected() -> None:
    app = create_app()
    with TestClient(app) as client:
        source_resp = client.post("/api/v1/source", json={"mode": "external"})
        assert source_resp.status_code == 200

        valid = {
            "schema_version": "0.1",
            "type": "driver.state",
            "session_id": "external-session",
            "source_id": "ext",
            "sequence": 1,
            "simulation_time_s": 0.1,
            "payload": {
                "gaze_region": "road",
                "attention": "attentive",
                "workload": 0.45,
                "confidence": 0.9,
                "available": True,
                "measurement_time_s": 0.1,
            },
        }
        ok = client.post("/api/v1/ingest", json=valid)
        assert ok.status_code == 200
        assert ok.json()["accepted"] is True

        wrong_schema = dict(valid)
        wrong_schema["schema_version"] = "9.9"
        rejected = client.post("/api/v1/ingest", json=wrong_schema)
        assert rejected.status_code == 422

        invalid_value = dict(valid)
        invalid_value["sequence"] = 2
        invalid_value["payload"] = dict(valid["payload"])
        invalid_value["payload"]["workload"] = 1.5
        invalid = client.post("/api/v1/ingest", json=invalid_value)
        assert invalid.status_code == 422


@pytest.mark.asyncio
async def test_seek_reconstructs_same_policy(tmp_path: Path) -> None:
    recording_manager = RecordingManager(tmp_path / "recordings")
    source_store = HMIStateStore()
    source_store.register_record_callback(recording_manager.record_event)
    source_store.register_update_callback(lambda *_args, **_kwargs: asyncio.sleep(0))

    snapshot = await source_store.snapshot()
    session = await recording_manager.start(snapshot.model_dump())
    for i in range(1, 6):
        envelope = DriverStateEnvelope(
            schema_version="0.1",
            type="driver.state",
            session_id="mock-session",
            source_id="mock",
            sequence=i,
            simulation_time_s=float(i),
            payload={
                "gaze_region": "road",
                "attention": "attentive",
                "workload": 0.8 if i >= 3 else 0.2,
                "confidence": 0.8,
                "available": True,
                "measurement_time_s": float(i),
            },
        )
        await source_store.ingest(envelope)
    await recording_manager.stop()

    replay_store = HMIStateStore()
    replay_store.register_update_callback(lambda *_args, **_kwargs: asyncio.sleep(0))
    replay_store.register_record_callback(None)
    replay = ReplaySourceEngine(replay_store, recording_manager)
    await replay.load_recording(session.recording_id)
    await replay.set_control(action="seek", seek_time_s=4.0)
    seek_state = await replay_store.snapshot()

    linear_store = HMIStateStore()
    adapter = TypeAdapter(DriverStateEnvelope)
    events = await recording_manager.read_events(session.recording_id)
    for entry in events:
        if entry["event_type"] != "input.normalized":
            continue
        payload = entry["payload"]["envelope"]
        if payload["type"] != "driver.state":
            continue
        if payload["simulation_time_s"] <= 4.0:
            await linear_store.ingest(adapter.validate_python(payload))
    linear_state = await linear_store.snapshot()

    assert seek_state.policy is not None
    assert linear_state.policy is not None
    assert seek_state.policy.model_dump(exclude={"policy_id", "input_sequences"}) == linear_state.policy.model_dump(
        exclude={"policy_id", "input_sequences"}
    )
