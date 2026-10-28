from __future__ import annotations

import asyncio
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, WebSocket
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from starlette.websockets import WebSocketDisconnect

from app.core.state import SequenceRejectedError
from app.protocol.models import (
    ExperimentCondition,
    InboundEnvelope,
    InteractionEventPayload,
    PreferenceProfile,
    SourceMode,
    UserPreferencesPayload,
)

router = APIRouter(prefix="/api/v1")
inbound_adapter = TypeAdapter(InboundEnvelope)


class PreferencesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: PreferenceProfile
    explanation_detail_level: Literal["short", "medium", "long"] = "medium"
    show_prediction_preference: bool = True


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    condition: ExperimentCondition


class SourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: SourceMode
    recording_id: str | None = None


class MockControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: Literal["city", "highway", "data_quality"] | None = None
    running: bool | None = None
    speed: float | None = Field(default=None, ge=0.1, le=5.0)
    workload: float | None = Field(default=None, ge=0.0, le=1.0)
    dms_available: bool | None = None
    attention: Literal["attentive", "distracted", "unknown"] | None = None
    gaze_region: Literal["road", "hmi", "elsewhere", "unknown"] | None = None
    stale_scene: bool | None = None
    disconnected: bool | None = None
    trigger_critical_event: bool | None = None
    trigger_noncritical_event: bool | None = None
    action: Literal["none", "reset"] = "none"


class InteractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_id: str
    action: Literal[
        "select_object",
        "clear_selection",
        "expand_explanation",
        "reduce_explanation",
        "set_profile",
        "set_manual_detail_request",
    ]
    value: str | bool | None = None


class ReplayControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["load", "play", "pause", "reset", "seek"]
    recording_id: str | None = None
    seek_time_s: float | None = Field(default=None, ge=0.0)
    speed: float | None = Field(default=None, ge=0.1, le=5.0)


@router.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@router.get("/schema")
async def schema() -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "envelope_schema": inbound_adapter.json_schema(),
        "websocket_contract": {
            "endpoint": "/api/v1/stream",
            "initial_message_type": "state.snapshot",
            "update_message_type": "state.update",
        },
    }


@router.get("/capabilities")
async def capabilities(request: Request) -> dict[str, Any]:
    state = await request.app.state.state_store.snapshot()
    return state.source_capabilities.model_dump()


@router.post("/ingest")
async def ingest(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    if request.app.state.current_source_mode != SourceMode.EXTERNAL:
        raise HTTPException(status_code=409, detail="ingest only allowed in external mode")
    try:
        envelope = inbound_adapter.validate_python(payload)
        await request.app.state.state_store.ingest(envelope)
    except SequenceRejectedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"accepted": True}


@router.put("/preferences")
async def put_preferences(request: Request, body: PreferencesRequest) -> dict[str, Any]:
    preferences = UserPreferencesPayload(
        profile=body.profile,
        explanation_detail_level=body.explanation_detail_level,
        show_prediction_preference=body.show_prediction_preference,
    )
    await request.app.state.state_store.set_preferences(preferences)
    return {"ok": True, "preferences": preferences.model_dump()}


@router.put("/experiment")
async def put_experiment(request: Request, body: ExperimentRequest) -> dict[str, Any]:
    await request.app.state.state_store.set_condition(body.condition)
    return {"ok": True, "condition": body.condition}


@router.post("/interactions")
async def post_interactions(request: Request, body: InteractionRequest) -> dict[str, Any]:
    event = InteractionEventPayload(
        interaction_id=f"int-{uuid4().hex[:8]}",
        display_id=body.display_id,
        action=body.action,
        value=body.value,
    )
    await request.app.state.state_store.apply_interaction(event)
    return {"ok": True}


@router.post("/source")
async def post_source(request: Request, body: SourceRequest) -> dict[str, Any]:
    if body.mode == SourceMode.REPLAY and not body.recording_id:
        raise HTTPException(status_code=400, detail="recording_id is required for replay mode")
    if body.mode == SourceMode.REPLAY:
        await request.app.state.replay_source.load_recording(body.recording_id)
        await request.app.state.state_store.set_source_mode(SourceMode.REPLAY, "replay")
    elif body.mode == SourceMode.EXTERNAL:
        await request.app.state.state_store.set_source_mode(SourceMode.EXTERNAL, "external")
    else:
        await request.app.state.state_store.set_source_mode(SourceMode.MOCK, "mock")
    request.app.state.current_source_mode = body.mode
    return {"ok": True, "mode": body.mode}


@router.post("/mock/control")
async def post_mock_control(request: Request, body: MockControlRequest) -> dict[str, Any]:
    if request.app.state.current_source_mode != SourceMode.MOCK:
        raise HTTPException(status_code=409, detail="mock control only available in mock mode")
    if body.action == "reset":
        await request.app.state.mock_source.reset()
    control = await request.app.state.mock_source.set_control(**body.model_dump(exclude_none=True))
    return {"ok": True, "control": control.__dict__}


@router.post("/recordings/start")
async def post_recording_start(request: Request) -> dict[str, Any]:
    state = await request.app.state.state_store.snapshot()
    session = await request.app.state.recording_manager.start(state.model_dump())
    return {"ok": True, "recording_id": session.recording_id}


@router.post("/recordings/stop")
async def post_recording_stop(request: Request) -> dict[str, Any]:
    session = await request.app.state.recording_manager.stop()
    return {"ok": True, "recording_id": session.recording_id if session else None}


@router.get("/recordings")
async def get_recordings(request: Request) -> dict[str, Any]:
    return {"recordings": await request.app.state.recording_manager.list_recordings()}


@router.post("/replay/control")
async def post_replay_control(request: Request, body: ReplayControlRequest) -> dict[str, Any]:
    if request.app.state.current_source_mode != SourceMode.REPLAY:
        raise HTTPException(status_code=409, detail="replay control only available in replay mode")
    if body.action == "load":
        if not body.recording_id:
            raise HTTPException(status_code=400, detail="recording_id required")
        await request.app.state.replay_source.load_recording(body.recording_id)
    control = await request.app.state.replay_source.set_control(
        action=body.action,
        seek_time_s=body.seek_time_s,
        speed=body.speed,
    )
    return {"ok": True, "state": control.__dict__}


@router.websocket("/stream")
async def stream(request: WebSocket) -> None:
    app_state = request.app.state
    initial_state = await app_state.state_store.snapshot()
    client = await app_state.stream_hub.connect(request, initial_state)
    sender_task = asyncio.create_task(app_state.stream_hub.client_sender_loop(client))
    try:
        while True:
            await request.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        await app_state.stream_hub.disconnect(client)
