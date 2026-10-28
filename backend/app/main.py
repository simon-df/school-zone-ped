from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.state import HMIStateStore
from app.core.stream import StreamHub
from app.protocol.models import SourceMode
from app.recording.manager import RecordingManager
from app.sources.mock_source import MockSourceEngine
from app.sources.replay_source import ReplaySourceEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    state_store = HMIStateStore()
    stream_hub = StreamHub()
    recording_manager = RecordingManager(
        base_dir=Path(__file__).resolve().parents[2] / "examples" / "recordings"
    )
    mock_source = MockSourceEngine(state_store)
    replay_source = ReplaySourceEngine(state_store, recording_manager)

    async def emit_update(state):
        await stream_hub.broadcast_state(state)
        await recording_manager.record_event("state.update", state.model_dump())

    state_store.register_update_callback(emit_update)
    state_store.register_record_callback(recording_manager.record_event)

    app.state.state_store = state_store
    app.state.stream_hub = stream_hub
    app.state.recording_manager = recording_manager
    app.state.mock_source = mock_source
    app.state.replay_source = replay_source
    app.state.current_source_mode = SourceMode.MOCK

    await mock_source.start()
    await replay_source.start()
    try:
        yield
    finally:
        await mock_source.stop()
        await replay_source.stop()


def create_app() -> FastAPI:
    app = FastAPI(title="Adaptive Automotive HMI Prototype", version="0.1", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
