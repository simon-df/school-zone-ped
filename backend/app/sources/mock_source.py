from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from app.core.scenarios import ScenarioId, generate_scenario_step
from app.core.state import HMIStateStore
from app.protocol.models import (
    AttentionState,
    AutomationStateEnvelope,
    DriverStateEnvelope,
    ExplanationEventEnvelope,
    GazeRegion,
    SceneSnapshotEnvelope,
    SourceMode,
    SystemStatusEnvelope,
    SystemStatusPayload,
)


@dataclass
class MockControl:
    scenario_id: ScenarioId = "city"
    running: bool = True
    speed: float = 1.0
    workload: float | None = 0.35
    dms_available: bool = True
    attention: AttentionState = AttentionState.ATTENTIVE
    gaze_region: GazeRegion = GazeRegion.ROAD
    stale_scene: bool = False
    disconnected: bool = False
    trigger_critical_event: bool = False
    trigger_noncritical_event: bool = False


class MockSourceEngine:
    def __init__(self, state_store: HMIStateStore) -> None:
        self._state_store = state_store
        self._control = MockControl()
        self._logical_time_s = 0.0
        self._tick_hz = 20.0
        self._task: asyncio.Task | None = None
        self._session_id = f"mock-session-{uuid4().hex[:6]}"
        self._sequence = {
            "scene.snapshot": 0,
            "driver.state": 0,
            "automation.state": 0,
            "explanation.event": 0,
            "system.status": 0,
        }

    @property
    def control(self) -> MockControl:
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

    async def reset(self) -> None:
        self._logical_time_s = 0.0
        self._session_id = f"mock-session-{uuid4().hex[:6]}"
        for key in self._sequence:
            self._sequence[key] = 0
        await self._state_store.reset_time_bound_state()

    async def set_control(self, **kwargs) -> MockControl:
        for key, value in kwargs.items():
            if value is not None and hasattr(self._control, key):
                setattr(self._control, key, value)
        return self._control

    async def _loop(self) -> None:
        tick_interval = 1.0 / self._tick_hz
        while True:
            await asyncio.sleep(tick_interval)
            if self._control.running:
                self._logical_time_s += tick_interval * self._control.speed

            if self._control.disconnected:
                await self._emit_status("disconnected")
                continue

            step = generate_scenario_step(
                scenario_id=self._control.scenario_id,
                simulation_time_s=self._logical_time_s,
                workload=self._control.workload,
                dms_available=self._control.dms_available,
                attention=self._control.attention,
                gaze_region=self._control.gaze_region,
                trigger_critical_event=self._control.trigger_critical_event,
                trigger_noncritical_event=self._control.trigger_noncritical_event,
                stale_scene=self._control.stale_scene,
            )
            await self._emit_status("connected")
            await self._state_store.ingest(
                SceneSnapshotEnvelope(
                    schema_version="0.1",
                    type="scene.snapshot",
                    session_id=self._session_id,
                    source_id="mock",
                    sequence=self._next("scene.snapshot"),
                    simulation_time_s=self._logical_time_s,
                    payload=step.scene,
                )
            )
            await self._state_store.ingest(
                DriverStateEnvelope(
                    schema_version="0.1",
                    type="driver.state",
                    session_id=self._session_id,
                    source_id="mock",
                    sequence=self._next("driver.state"),
                    simulation_time_s=self._logical_time_s,
                    payload=step.driver,
                )
            )
            await self._state_store.ingest(
                AutomationStateEnvelope(
                    schema_version="0.1",
                    type="automation.state",
                    session_id=self._session_id,
                    source_id="mock",
                    sequence=self._next("automation.state"),
                    simulation_time_s=self._logical_time_s,
                    payload=step.automation,
                )
            )
            if step.explanation is not None:
                await self._state_store.ingest(
                    ExplanationEventEnvelope(
                        schema_version="0.1",
                        type="explanation.event",
                        session_id=self._session_id,
                        source_id="mock",
                        sequence=self._next("explanation.event"),
                        simulation_time_s=self._logical_time_s,
                        payload=step.explanation,
                    )
                )
            self._control.trigger_critical_event = False
            self._control.trigger_noncritical_event = False

    async def _emit_status(self, connection_state: Literal["connected", "disconnected", "paused"]) -> None:
        await self._state_store.ingest(
            SystemStatusEnvelope(
                schema_version="0.1",
                type="system.status",
                session_id=self._session_id,
                source_id="mock",
                sequence=self._next("system.status"),
                simulation_time_s=self._logical_time_s,
                payload=SystemStatusPayload(
                    source_mode=SourceMode.MOCK,
                    source_id="mock",
                    data_connection=connection_state,
                    scene_data_stale=self._control.stale_scene,
                    scene_age_s=0.0,
                    replay_paused=False,
                ),
            )
        )

    def _next(self, message_type: str) -> int:
        self._sequence[message_type] += 1
        return self._sequence[message_type]


import contextlib  # noqa: E402  (kept near usage for readability)
