from __future__ import annotations

import asyncio
import copy
import time
from typing import Callable, Coroutine
from uuid import uuid4

from app.core.adaptation import DecisionResult, PolicyRuntimeState, evaluate_policy
from app.protocol.models import (
    Capabilities,
    CanonicalState,
    ExperimentCondition,
    InboundEnvelope,
    InteractionEventPayload,
    PreferenceProfile,
    SourceMode,
    SystemStatusPayload,
    UserPreferencesPayload,
)

UpdateCallback = Callable[[CanonicalState], Coroutine[None, None, None]]
RecordCallback = Callable[[str, dict], Coroutine[None, None, None]]


class SequenceRejectedError(ValueError):
    pass


class HMIStateStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._policy_runtime_state = PolicyRuntimeState()
        self._record_callback: RecordCallback | None = None
        self._update_callback: UpdateCallback | None = None
        self._sequence_tracker: dict[tuple[str, str], tuple[str, int]] = {}
        self._critical_event_until_s: float = 0.0
        self._last_scene_sim_time_s: float | None = None
        self._state = CanonicalState(
            session_id=f"session-{uuid4().hex[:8]}",
            source_mode=SourceMode.MOCK,
            condition=ExperimentCondition.STATIC,
            preferences=UserPreferencesPayload(
                profile=PreferenceProfile.SAFETY,
                explanation_detail_level="medium",
                show_prediction_preference=True,
            ),
            system_status=SystemStatusPayload(
                source_mode=SourceMode.MOCK,
                source_id="mock",
                data_connection="connected",
                scene_data_stale=False,
                scene_age_s=0.0,
                replay_paused=False,
            ),
            source_capabilities=Capabilities(
                source_id="mock",
                supports_objects=True,
                supports_lanes=True,
                supports_prediction=True,
                supports_gaze=True,
            ),
        )

    def register_update_callback(self, callback: UpdateCallback) -> None:
        self._update_callback = callback

    def register_record_callback(self, callback: RecordCallback | None) -> None:
        self._record_callback = callback

    async def snapshot(self) -> CanonicalState:
        async with self._lock:
            return copy.deepcopy(self._state)

    async def reset_time_bound_state(self) -> None:
        async with self._lock:
            self._policy_runtime_state = PolicyRuntimeState()
            self._critical_event_until_s = 0.0
            self._last_scene_sim_time_s = None
            self._state.active_explanations.clear()
            self._state.selected_track_id = None
            self._state.manual_detail_requested = False
            self._state.policy = None
            self._state.last_simulation_time_s = 0.0
            self._state.system_status.scene_age_s = 0.0
            self._state.system_status.scene_data_stale = False
            self._sequence_tracker.clear()
        await self._emit_update()

    async def set_source_mode(self, source_mode: SourceMode, source_id: str) -> None:
        async with self._lock:
            self._state.source_mode = source_mode
            self._state.system_status.source_mode = source_mode
            self._state.system_status.source_id = source_id
            self._state.system_status.data_connection = (
                "paused" if source_mode == SourceMode.REPLAY else "connected"
            )
            self._state.source_capabilities = Capabilities(
                source_id=source_id,
                supports_objects=True,
                supports_lanes=True,
                supports_prediction=source_mode != SourceMode.EXTERNAL or True,
                supports_gaze=source_mode != SourceMode.EXTERNAL or True,
            )
            self._policy_runtime_state = PolicyRuntimeState()
        await self._emit_update()

    async def set_condition(self, condition: ExperimentCondition) -> None:
        async with self._lock:
            self._state.condition = condition
            self._recompute_policy_locked(self._state.last_simulation_time_s)
        await self._record("experiment.condition", {"condition": condition.value})
        await self._emit_update()

    async def set_preferences(self, preferences: UserPreferencesPayload) -> None:
        async with self._lock:
            self._state.preferences = preferences
            self._recompute_policy_locked(self._state.last_simulation_time_s)
        await self._record("preferences.updated", preferences.model_dump())
        await self._emit_update()

    async def apply_interaction(self, interaction: InteractionEventPayload) -> None:
        async with self._lock:
            if interaction.action == "select_object" and isinstance(interaction.value, str):
                self._state.selected_track_id = interaction.value
            elif interaction.action == "clear_selection":
                self._state.selected_track_id = None
            elif interaction.action == "expand_explanation":
                self._state.manual_detail_requested = True
            elif interaction.action == "reduce_explanation":
                self._state.manual_detail_requested = False
            elif interaction.action == "set_profile" and isinstance(interaction.value, str):
                self._state.preferences.profile = PreferenceProfile(interaction.value)
            elif interaction.action == "set_manual_detail_request":
                self._state.manual_detail_requested = bool(interaction.value)
            self._recompute_policy_locked(self._state.last_simulation_time_s)
        await self._record("interaction.event", interaction.model_dump())
        await self._emit_update()

    async def ingest(self, envelope: InboundEnvelope) -> None:
        now_monotonic = time.monotonic()
        async with self._lock:
            self._validate_sequence_locked(envelope)
            logical_time = envelope.simulation_time_s
            self._state.last_simulation_time_s = max(self._state.last_simulation_time_s, logical_time)
            if envelope.type == "scene.snapshot":
                self._state.scene = envelope.payload
                self._last_scene_sim_time_s = logical_time
                current_track_ids = {track.track_id for track in envelope.payload.tracks}
                if self._state.selected_track_id and self._state.selected_track_id not in current_track_ids:
                    self._state.selected_track_id = None
            elif envelope.type == "driver.state":
                self._state.driver_state = envelope.payload
            elif envelope.type == "automation.state":
                self._state.automation_state = envelope.payload
            elif envelope.type == "explanation.event":
                self._state.active_explanations.append(envelope.payload)
                self._state.active_explanations = self._state.active_explanations[-15:]
                if envelope.payload.critical:
                    self._critical_event_until_s = max(self._critical_event_until_s, logical_time + 4.0)
            elif envelope.type == "user.preferences":
                self._state.preferences = envelope.payload
            elif envelope.type == "interaction.event":
                interaction = envelope.payload
                if interaction.action == "select_object" and isinstance(interaction.value, str):
                    self._state.selected_track_id = interaction.value
                elif interaction.action == "clear_selection":
                    self._state.selected_track_id = None
                elif interaction.action == "expand_explanation":
                    self._state.manual_detail_requested = True
                elif interaction.action == "reduce_explanation":
                    self._state.manual_detail_requested = False
                elif interaction.action == "set_profile" and isinstance(interaction.value, str):
                    self._state.preferences.profile = PreferenceProfile(interaction.value)
                elif interaction.action == "set_manual_detail_request":
                    self._state.manual_detail_requested = bool(interaction.value)
            elif envelope.type == "system.status":
                self._state.system_status = envelope.payload

            self._state.system_status.scene_age_s = (
                max(0.0, logical_time - self._last_scene_sim_time_s) if self._last_scene_sim_time_s is not None else 0.0
            )
            self._state.system_status.scene_data_stale = self._state.system_status.scene_age_s > 1.0
            if self._state.source_mode == SourceMode.REPLAY:
                self._state.system_status.replay_paused = self._state.system_status.data_connection == "paused"

            self._recompute_policy_locked(logical_time)
            if envelope.type in {"scene.snapshot", "driver.state", "automation.state", "explanation.event"}:
                await self._record(
                    "input.normalized",
                    {"received_monotonic_s": now_monotonic, "envelope": envelope.model_dump()},
                )
        await self._emit_update()

    def _validate_sequence_locked(self, envelope: InboundEnvelope) -> None:
        key = (envelope.source_id, envelope.type)
        previous = self._sequence_tracker.get(key)
        if previous is None:
            self._sequence_tracker[key] = (envelope.session_id, envelope.sequence)
            return
        previous_session, previous_sequence = previous
        if previous_session != envelope.session_id:
            self._sequence_tracker[key] = (envelope.session_id, envelope.sequence)
            return
        if envelope.sequence <= previous_sequence:
            raise SequenceRejectedError(
                f"stale or duplicate sequence {envelope.sequence} for {envelope.type} "
                f"(latest={previous_sequence})"
            )
        self._sequence_tracker[key] = (envelope.session_id, envelope.sequence)

    def _recompute_policy_locked(self, logical_time_s: float) -> DecisionResult:
        decision = evaluate_policy(
            condition=self._state.condition,
            driver_state=self._state.driver_state,
            user_preferences=self._state.preferences,
            previous_policy_state=self._policy_runtime_state,
            logical_time_s=logical_time_s,
            has_critical_event=logical_time_s <= self._critical_event_until_s,
            manual_detail_requested=self._state.manual_detail_requested,
            scene_stale=self._state.system_status.scene_data_stale,
            input_sequences={f"{key[0]}::{key[1]}": value[1] for key, value in self._sequence_tracker.items()},
        )
        if self._state.selected_track_id:
            decision.policy.highlighted_track_ids = [self._state.selected_track_id]
        self._policy_runtime_state = decision.runtime_state
        self._state.policy = decision.policy
        return decision

    async def _emit_update(self) -> None:
        callback = self._update_callback
        if callback is None:
            return
        await callback(await self.snapshot())

    async def _record(self, event_type: str, payload: dict) -> None:
        callback = self._record_callback
        if callback is None:
            return
        await callback(event_type, payload)
