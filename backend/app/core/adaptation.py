from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from app.protocol.models import (
    AttentionState,
    DriverStatePayload,
    ExperimentCondition,
    InformationDensity,
    PreferenceProfile,
    PresentationPolicyPayload,
    UserPreferencesPayload,
)

RULE_VERSION = "ruleset-0.1"


@dataclass
class PolicyRuntimeState:
    density: InformationDensity = InformationDensity.MEDIUM
    candidate_density: InformationDensity = InformationDensity.MEDIUM
    candidate_since_s: float = 0.0
    last_density_change_s: float = 0.0


@dataclass
class DecisionResult:
    policy: PresentationPolicyPayload
    runtime_state: PolicyRuntimeState
    decision_events: list[str] = field(default_factory=list)


HIGH_WORKLOAD_THRESHOLD = 0.7
LOW_WORKLOAD_THRESHOLD = 0.5
STABILITY_SECONDS = 1.0
MIN_DWELL_SECONDS = 3.0


def _base_reason_codes(condition: ExperimentCondition, driver_known: bool) -> list[str]:
    reasons: list[str] = []
    if condition == ExperimentCondition.STATIC:
        reasons.append("STATIC_BASELINE")
    if not driver_known:
        reasons.append("DRIVER_STATE_UNKNOWN")
    return reasons


def _determine_candidate_density(
    condition: ExperimentCondition,
    driver_state: DriverStatePayload | None,
) -> tuple[InformationDensity, list[str]]:
    reasons: list[str] = []
    if condition == ExperimentCondition.STATIC:
        return InformationDensity.MEDIUM, reasons

    if driver_state is None or not driver_state.available or driver_state.attention == AttentionState.UNKNOWN:
        reasons.append("DRIVER_STATE_UNKNOWN")
        return InformationDensity.MEDIUM, reasons

    workload = driver_state.workload
    if workload is None:
        reasons.append("DRIVER_STATE_UNKNOWN")
        return InformationDensity.MEDIUM, reasons

    if workload >= HIGH_WORKLOAD_THRESHOLD:
        reasons.append("HIGH_WORKLOAD")
        return InformationDensity.LOW, reasons
    if workload <= LOW_WORKLOAD_THRESHOLD:
        return InformationDensity.MEDIUM, reasons
    return InformationDensity.MEDIUM, reasons


def evaluate_policy(
    *,
    condition: ExperimentCondition,
    driver_state: DriverStatePayload | None,
    user_preferences: UserPreferencesPayload,
    previous_policy_state: PolicyRuntimeState | None,
    logical_time_s: float,
    has_critical_event: bool,
    manual_detail_requested: bool,
    scene_stale: bool,
    input_sequences: dict[str, int],
) -> DecisionResult:
    runtime = previous_policy_state or PolicyRuntimeState()

    candidate, candidate_reasons = _determine_candidate_density(condition, driver_state)
    reasons = _base_reason_codes(condition, driver_state is not None and driver_state.available)
    reasons.extend([reason for reason in candidate_reasons if reason not in reasons])

    if runtime.candidate_density != candidate:
        runtime.candidate_density = candidate
        runtime.candidate_since_s = logical_time_s

    candidate_stable = logical_time_s - runtime.candidate_since_s >= STABILITY_SECONDS
    dwell_elapsed = logical_time_s - runtime.last_density_change_s >= MIN_DWELL_SECONDS

    if candidate_stable and dwell_elapsed and runtime.density != candidate:
        runtime.density = candidate
        runtime.last_density_change_s = logical_time_s

    show_prediction = True
    show_uncertainty = runtime.density != InformationDensity.LOW
    show_object_labels = runtime.density != InformationDensity.LOW
    explanation_level = "medium"

    if runtime.density == InformationDensity.LOW:
        explanation_level = "short"

    if user_preferences.profile == PreferenceProfile.TECH:
        if (
            driver_state
            and driver_state.available
            and driver_state.workload is not None
            and driver_state.workload <= LOW_WORKLOAD_THRESHOLD
            and runtime.density != InformationDensity.LOW
        ):
            show_prediction = True
            show_uncertainty = True
            reasons.append("TECH_PROFILE_DETAIL")
    elif user_preferences.profile == PreferenceProfile.CONTROL:
        explanation_level = "long" if runtime.density != InformationDensity.LOW else "medium"

    if manual_detail_requested:
        reasons.append("MANUAL_DETAIL_REQUEST")
        if runtime.density != InformationDensity.LOW:
            explanation_level = "long"

    if runtime.density == InformationDensity.LOW and "HIGH_WORKLOAD" in reasons:
        # Rule priority: high workload caps details even if profile would ask for more.
        explanation_level = "short"
        show_object_labels = False
        show_uncertainty = False

    if has_critical_event:
        reasons.append("CRITICAL_EVENT")

    if scene_stale:
        reasons.append("SCENE_DATA_STALE")
        show_prediction = False

    reason_codes: list[str] = []
    for code in reasons:
        if code not in reason_codes:
            reason_codes.append(code)

    policy = PresentationPolicyPayload(
        policy_id=f"policy-{uuid4().hex[:8]}",
        rule_version=RULE_VERSION,
        information_density=runtime.density,
        show_object_labels=show_object_labels,
        show_prediction=show_prediction,
        show_uncertainty=show_uncertainty,
        highlighted_track_ids=[],
        explanation_detail_by_display={
            "center": explanation_level,
            "panorama": "short",
            "hud": "short",
            "workbench": "long",
        },
        reason_codes=reason_codes,
        manual_detail_requested=manual_detail_requested,
        input_sequences=input_sequences,
    )
    return DecisionResult(policy=policy, runtime_state=runtime, decision_events=reason_codes)
