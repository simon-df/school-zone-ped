from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter

from app.core.state import HMIStateStore
from app.protocol.models import InboundEnvelope, SourceMode, SystemStatusEnvelope, SystemStatusPayload
from app.recording.manager import RecordingManager


@dataclass
class ReplayControlState:
    recording_id: str | None = None
    running: bool = False
    speed: float = 1.0
    logical_time_s: float = 0.0
    cursor: int = 0


class ReplaySourceEngine:
    def __init__(self, state_store: HMIStateStore, recording_manager: RecordingManager) -> None:
        self._state_store = state_store
        self._recording_manager = recording_manager
        self._control = ReplayControlState()
        self._events: list[dict[str, Any]] = []
        self._task: asyncio.Task | None = None
        self._session_id = "replay-session"
        self._status_seq = 0
        self._adapter = TypeAdapter(InboundEnvelope)

    @property
    def control(self) -> ReplayControlState:
        return self._control

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def load_recording(self, recording_id: str) -> None:
        events = await self._recording_manager.read_events(recording_id)
        normalized = [
            entry["payload"]["envelope"]
            for entry in events
            if entry.get("event_type") == "input.normalized"
        ]
        self._events = sorted(normalized, key=lambda event: event.get("simulation_time_s", 0.0))
        self._control.recording_id = recording_id
        self._control.cursor = 0
        self._control.logical_time_s = 0.0
        self._control.running = False
        await self._state_store.reset_time_bound_state()

    async def set_control(
        self,
        *,
        action: str,
        seek_time_s: float | None = None,
        speed: float | None = None,
    ) -> ReplayControlState:
        if speed is not None:
            self._control.speed = max(0.1, min(speed, 5.0))
        if action == "play":
            self._control.running = True
        elif action == "pause":
            self._control.running = False
        elif action == "reset":
            self._control.running = False
            await self._rebuild_until(0.0)
        elif action == "seek":
            self._control.running = False
            await self._rebuild_until(seek_time_s or 0.0)
        return self._control

    async def _loop(self) -> None:
        tick_s = 1.0 / 20.0
        while True:
            await asyncio.sleep(tick_s)
            if not self._control.running:
                await self._emit_status("paused")
                continue
            if not self._events:
                await self._emit_status("paused")
                self._control.running = False
                continue

            self._control.logical_time_s += tick_s * self._control.speed
            await self._emit_status("connected")
            while self._control.cursor < len(self._events):
                event = self._events[self._control.cursor]
                event_time = float(event.get("simulation_time_s", 0.0))
                if event_time > self._control.logical_time_s:
                    break
                await self._ingest_recorded(event)
                self._control.cursor += 1
            if self._control.cursor >= len(self._events):
                self._control.running = False

    async def _rebuild_until(self, target_time_s: float) -> None:
        await self._state_store.reset_time_bound_state()
        self._control.cursor = 0
        self._control.logical_time_s = target_time_s
        while self._control.cursor < len(self._events):
            event = self._events[self._control.cursor]
            event_time = float(event.get("simulation_time_s", 0.0))
            if event_time > target_time_s:
                break
            await self._ingest_recorded(event)
            self._control.cursor += 1
        await self._emit_status("paused")

    async def _ingest_recorded(self, payload: dict[str, Any]) -> None:
        envelope = self._adapter.validate_python(payload)
        await self._state_store.ingest(envelope)

    async def _emit_status(self, connection_state: str) -> None:
        self._status_seq += 1
        await self._state_store.ingest(
            SystemStatusEnvelope(
                schema_version="0.1",
                type="system.status",
                session_id=self._session_id,
                source_id="replay",
                sequence=self._status_seq,
                simulation_time_s=self._control.logical_time_s,
                payload=SystemStatusPayload(
                    source_mode=SourceMode.REPLAY,
                    source_id="replay",
                    data_connection="paused" if connection_state == "paused" else "connected",
                    scene_data_stale=False,
                    scene_age_s=0.0,
                    replay_paused=connection_state == "paused",
                ),
            )
        )


import contextlib  # noqa: E402
