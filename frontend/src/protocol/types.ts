export type InformationDensity = 'low' | 'medium' | 'high'
export type SourceMode = 'mock' | 'replay' | 'external'
export type PreferenceProfile = 'safety' | 'efficiency' | 'control' | 'tech'

export interface Point3D {
  x_m: number
  y_m: number
  z_m: number
}

export interface ObjectTrack {
  track_id: string
  object_class: 'vehicle' | 'pedestrian' | 'bicycle' | 'unknown'
  position: Point3D
  yaw_rad: number
  extent_m: Point3D
  velocity_mps: Point3D
  existence_probability?: number
  predicted_trajectories?: Array<{
    hypothesis_id: string
    probability: number
    points: Point3D[]
  }>
}

export interface SceneSnapshotPayload {
  ego: {
    speed_mps: number
    pose: Point3D
    yaw_rad: number
  }
  tracks: ObjectTrack[]
  lanes: {
    lane_id: string
    points: Point3D[]
  }[]
  ego_plan: {
    maneuver: string
    trajectory: Point3D[]
  }
  provenance: 'synthetic' | 'ground_truth' | 'perception' | 'fusion'
}

export interface DriverStatePayload {
  gaze_region: 'road' | 'hmi' | 'elsewhere' | 'unknown'
  attention: 'attentive' | 'distracted' | 'unknown'
  workload: number | null
  confidence: number
  available: boolean
  measurement_time_s: number
}

export interface ExplanationEventPayload {
  event_id: string
  title: string
  text: string
  referenced_track_id: string | null
  critical: boolean
}

export interface PresentationPolicyPayload {
  policy_id: string
  rule_version: string
  information_density: InformationDensity
  show_object_labels: boolean
  show_prediction: boolean
  show_uncertainty: boolean
  highlighted_track_ids: string[]
  explanation_detail_by_display: Record<string, 'short' | 'medium' | 'long'>
  reason_codes: string[]
  manual_detail_requested: boolean
  input_sequences: Record<string, number>
}

export interface CanonicalState {
  schema_version: '0.1'
  session_id: string
  source_mode: SourceMode
  condition: 'static' | 'adaptive'
  preferences: {
    profile: PreferenceProfile
    explanation_detail_level: 'short' | 'medium' | 'long'
    show_prediction_preference: boolean
  }
  scene: SceneSnapshotPayload | null
  driver_state: DriverStatePayload | null
  automation_state: {
    mode: string
    engaged: boolean
    confidence: number
  } | null
  active_explanations: ExplanationEventPayload[]
  selected_track_id: string | null
  manual_detail_requested: boolean
  policy: PresentationPolicyPayload | null
  system_status: {
    source_mode: SourceMode
    source_id: string
    data_connection: 'connected' | 'disconnected' | 'paused'
    scene_data_stale: boolean
    scene_age_s: number
    replay_paused: boolean
  }
  last_simulation_time_s: number
  source_capabilities: {
    source_id: string
    supports_objects: boolean
    supports_lanes: boolean
    supports_prediction: boolean
    supports_gaze: boolean
  }
}

export interface StateSnapshotMessage {
  type: 'state.snapshot'
  state: CanonicalState
}

export interface StateUpdateMessage {
  type: 'state.update'
  state: CanonicalState
  changed_at_simulation_time_s: number
}
