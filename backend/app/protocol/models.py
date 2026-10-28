from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "0.1"


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObjectClass(str, Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    BICYCLE = "bicycle"
    UNKNOWN = "unknown"


class Provenance(str, Enum):
    SYNTHETIC = "synthetic"
    GROUND_TRUTH = "ground_truth"
    PERCEPTION = "perception"
    FUSION = "fusion"


class GazeRegion(str, Enum):
    ROAD = "road"
    HMI = "hmi"
    ELSEWHERE = "elsewhere"
    UNKNOWN = "unknown"


class AttentionState(str, Enum):
    ATTENTIVE = "attentive"
    DISTRACTED = "distracted"
    UNKNOWN = "unknown"


class InformationDensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ExperimentCondition(str, Enum):
    STATIC = "static"
    ADAPTIVE = "adaptive"


class PreferenceProfile(str, Enum):
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    CONTROL = "control"
    TECH = "tech"


class SourceMode(str, Enum):
    MOCK = "mock"
    REPLAY = "replay"
    EXTERNAL = "external"


class Point3D(StrictBaseModel):
    x_m: float
    y_m: float
    z_m: float = 0.0

    @field_validator("x_m", "y_m", "z_m")
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not float("-inf") < value < float("inf"):
            raise ValueError("point coordinate must be finite")
        return value


class Covariance2D(StrictBaseModel):
    xx_m2: float = Field(ge=0.0)
    xy_m2: float
    yy_m2: float = Field(ge=0.0)


class PredictedTrajectory(StrictBaseModel):
    hypothesis_id: str
    probability: float = Field(ge=0.0, le=1.0)
    points: list[Point3D]


class LanePolyline(StrictBaseModel):
    lane_id: str
    points: list[Point3D]


class EgoState(StrictBaseModel):
    speed_mps: float = Field(ge=0.0)
    pose: Point3D
    yaw_rad: float


class ObjectTrack(StrictBaseModel):
    track_id: str
    object_class: ObjectClass
    position: Point3D
    yaw_rad: float
    extent_m: Point3D
    velocity_mps: Point3D
    existence_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    predicted_trajectories: list[PredictedTrajectory] | None = None
    covariance_2d_m2: Covariance2D | None = None


class EgoPlan(StrictBaseModel):
    maneuver: str
    trajectory: list[Point3D]


class SceneSnapshotPayload(StrictBaseModel):
    ego: EgoState
    tracks: list[ObjectTrack]
    lanes: list[LanePolyline]
    ego_plan: EgoPlan
    provenance: Provenance


class DriverStatePayload(StrictBaseModel):
    gaze_region: GazeRegion
    attention: AttentionState
    workload: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    available: bool
    measurement_time_s: float = Field(ge=0.0)


class AutomationStatePayload(StrictBaseModel):
    mode: str
    engaged: bool
    confidence: float = Field(ge=0.0, le=1.0)


class ExplanationEventPayload(StrictBaseModel):
    event_id: str
    title: str
    text: str
    referenced_track_id: str | None = None
    critical: bool = False


class UserPreferencesPayload(StrictBaseModel):
    profile: PreferenceProfile
    explanation_detail_level: Literal["short", "medium", "long"] = "medium"
    show_prediction_preference: bool = True


class PresentationPolicyPayload(StrictBaseModel):
    policy_id: str
    rule_version: str
    information_density: InformationDensity
    show_object_labels: bool
    show_prediction: bool
    show_uncertainty: bool
    highlighted_track_ids: list[str]
    explanation_detail_by_display: dict[str, Literal["short", "medium", "long"]]
    reason_codes: list[str]
    manual_detail_requested: bool
    input_sequences: dict[str, int]


class InteractionEventPayload(StrictBaseModel):
    interaction_id: str
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


class SystemStatusPayload(StrictBaseModel):
    source_mode: SourceMode
    source_id: str
    data_connection: Literal["connected", "disconnected", "paused"]
    scene_data_stale: bool
    scene_age_s: float = Field(ge=0.0)
    replay_paused: bool


class EnvelopeBase(StrictBaseModel):
    schema_version: Literal[SCHEMA_VERSION]
    type: str
    session_id: str
    source_id: str
    sequence: int = Field(ge=0)
    simulation_time_s: float = Field(ge=0.0)


class SceneSnapshotEnvelope(EnvelopeBase):
    type: Literal["scene.snapshot"]
    payload: SceneSnapshotPayload


class DriverStateEnvelope(EnvelopeBase):
    type: Literal["driver.state"]
    payload: DriverStatePayload


class AutomationStateEnvelope(EnvelopeBase):
    type: Literal["automation.state"]
    payload: AutomationStatePayload


class ExplanationEventEnvelope(EnvelopeBase):
    type: Literal["explanation.event"]
    payload: ExplanationEventPayload


class UserPreferencesEnvelope(EnvelopeBase):
    type: Literal["user.preferences"]
    payload: UserPreferencesPayload


class PresentationPolicyEnvelope(EnvelopeBase):
    type: Literal["presentation.policy"]
    payload: PresentationPolicyPayload


class InteractionEventEnvelope(EnvelopeBase):
    type: Literal["interaction.event"]
    payload: InteractionEventPayload


class SystemStatusEnvelope(EnvelopeBase):
    type: Literal["system.status"]
    payload: SystemStatusPayload


InboundEnvelope = Annotated[
    SceneSnapshotEnvelope
    | DriverStateEnvelope
    | AutomationStateEnvelope
    | ExplanationEventEnvelope
    | UserPreferencesEnvelope
    | InteractionEventEnvelope
    | SystemStatusEnvelope,
    Field(discriminator="type"),
]


class Capabilities(StrictBaseModel):
    source_id: str
    supports_objects: bool
    supports_lanes: bool
    supports_prediction: bool
    supports_gaze: bool


class CanonicalState(StrictBaseModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    session_id: str
    source_mode: SourceMode
    condition: ExperimentCondition
    preferences: UserPreferencesPayload
    scene: SceneSnapshotPayload | None = None
    driver_state: DriverStatePayload | None = None
    automation_state: AutomationStatePayload | None = None
    active_explanations: list[ExplanationEventPayload] = Field(default_factory=list)
    selected_track_id: str | None = None
    manual_detail_requested: bool = False
    policy: PresentationPolicyPayload | None = None
    system_status: SystemStatusPayload
    last_simulation_time_s: float = 0.0
    source_capabilities: Capabilities


class StateSnapshotMessage(StrictBaseModel):
    type: Literal["state.snapshot"] = "state.snapshot"
    state: CanonicalState


class StateUpdateMessage(StrictBaseModel):
    type: Literal["state.update"] = "state.update"
    state: CanonicalState
    changed_at_simulation_time_s: float
