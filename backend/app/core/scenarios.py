from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.protocol.models import (
    AttentionState,
    AutomationStatePayload,
    DriverStatePayload,
    EgoPlan,
    EgoState,
    ExplanationEventPayload,
    GazeRegion,
    LanePolyline,
    ObjectClass,
    ObjectTrack,
    Point3D,
    PredictedTrajectory,
    Provenance,
    SceneSnapshotPayload,
)

ScenarioId = Literal["city", "highway", "data_quality"]


@dataclass
class ScenarioStep:
    scene: SceneSnapshotPayload
    driver: DriverStatePayload
    automation: AutomationStatePayload
    explanation: ExplanationEventPayload | None = None


def _base_lanes() -> list[LanePolyline]:
    return [
        LanePolyline(
            lane_id="lane_left",
            points=[Point3D(x_m=x, y_m=2.0, z_m=0.0) for x in range(-20, 81, 10)],
        ),
        LanePolyline(
            lane_id="lane_right",
            points=[Point3D(x_m=x, y_m=-2.0, z_m=0.0) for x in range(-20, 81, 10)],
        ),
    ]


def generate_scenario_step(
    scenario_id: ScenarioId,
    simulation_time_s: float,
    workload: float | None,
    dms_available: bool,
    attention: AttentionState,
    gaze_region: GazeRegion,
    trigger_critical_event: bool,
    trigger_noncritical_event: bool,
    stale_scene: bool,
) -> ScenarioStep:
    if scenario_id == "city":
        return _city_step(
            simulation_time_s=simulation_time_s,
            workload=workload,
            dms_available=dms_available,
            attention=attention,
            gaze_region=gaze_region,
            trigger_critical_event=trigger_critical_event,
            trigger_noncritical_event=trigger_noncritical_event,
            stale_scene=stale_scene,
        )
    if scenario_id == "highway":
        return _highway_step(
            simulation_time_s=simulation_time_s,
            workload=workload,
            dms_available=dms_available,
            attention=attention,
            gaze_region=gaze_region,
            trigger_critical_event=trigger_critical_event,
            trigger_noncritical_event=trigger_noncritical_event,
            stale_scene=stale_scene,
        )
    return _data_quality_step(
        simulation_time_s=simulation_time_s,
        workload=workload,
        dms_available=dms_available,
        attention=attention,
        gaze_region=gaze_region,
        trigger_critical_event=trigger_critical_event,
        trigger_noncritical_event=trigger_noncritical_event,
        stale_scene=stale_scene,
    )


def _city_step(
    *,
    simulation_time_s: float,
    workload: float | None,
    dms_available: bool,
    attention: AttentionState,
    gaze_region: GazeRegion,
    trigger_critical_event: bool,
    trigger_noncritical_event: bool,
    stale_scene: bool,
) -> ScenarioStep:
    pedestrian_x = max(8.0 - simulation_time_s * 1.2, 0.3)
    track = ObjectTrack(
        track_id="ped-1",
        object_class=ObjectClass.PEDESTRIAN,
        position=Point3D(x_m=pedestrian_x, y_m=1.2, z_m=0.0),
        yaw_rad=0.0,
        extent_m=Point3D(x_m=0.6, y_m=0.6, z_m=1.7),
        velocity_mps=Point3D(x_m=-1.2, y_m=0.0, z_m=0.0),
        existence_probability=0.97,
        predicted_trajectories=[
            PredictedTrajectory(
                hypothesis_id="cross",
                probability=0.78,
                points=[
                    Point3D(x_m=pedestrian_x - i * 0.6, y_m=1.2 - i * 0.25, z_m=0.0)
                    for i in range(1, 6)
                ],
            )
        ],
    )
    ego_plan = EgoPlan(
        maneuver="wait_for_pedestrian" if pedestrian_x < 4.2 else "proceed",
        trajectory=[Point3D(x_m=i, y_m=0.0, z_m=0.0) for i in range(0, 18, 3)],
    )
    scene = SceneSnapshotPayload(
        ego=EgoState(speed_mps=8.5 if pedestrian_x > 5.0 else 2.0, pose=Point3D(x_m=0.0, y_m=0.0, z_m=0.0), yaw_rad=0.0),
        tracks=[track],
        lanes=_base_lanes(),
        ego_plan=ego_plan,
        provenance=Provenance.FUSION,
    )
    driver = DriverStatePayload(
        gaze_region=gaze_region if dms_available else GazeRegion.UNKNOWN,
        attention=attention if dms_available else AttentionState.UNKNOWN,
        workload=workload if dms_available else None,
        confidence=0.9 if dms_available else 0.0,
        available=dms_available,
        measurement_time_s=simulation_time_s if not stale_scene else max(simulation_time_s - 3.0, 0.0),
    )
    automation = AutomationStatePayload(mode="city_assist", engaged=True, confidence=0.88)
    explanation = None
    if trigger_critical_event:
        explanation = ExplanationEventPayload(
            event_id=f"crit-{int(simulation_time_s*10)}",
            title="Kritisches Ereignis",
            text="Fußgänger kreuzt erwartete Trajektorie; Plan wurde auf Warten gesetzt.",
            referenced_track_id="ped-1",
            critical=True,
        )
    elif trigger_noncritical_event or 3.0 < simulation_time_s < 3.2:
        explanation = ExplanationEventPayload(
            event_id=f"info-{int(simulation_time_s*10)}",
            title="Planungsanpassung",
            text="Ego-Plan priorisiert Komfortbremsung vor Querung.",
            referenced_track_id="ped-1",
            critical=False,
        )
    return ScenarioStep(scene=scene, driver=driver, automation=automation, explanation=explanation)


def _highway_step(
    *,
    simulation_time_s: float,
    workload: float | None,
    dms_available: bool,
    attention: AttentionState,
    gaze_region: GazeRegion,
    trigger_critical_event: bool,
    trigger_noncritical_event: bool,
    stale_scene: bool,
) -> ScenarioStep:
    lead_x = 18 - simulation_time_s * 0.5
    adjacent_x = 12 + simulation_time_s * 0.3
    tracks = [
        ObjectTrack(
            track_id="veh-lead",
            object_class=ObjectClass.VEHICLE,
            position=Point3D(x_m=lead_x, y_m=0.0, z_m=0.0),
            yaw_rad=0.0,
            extent_m=Point3D(x_m=4.2, y_m=1.8, z_m=1.6),
            velocity_mps=Point3D(x_m=-0.5, y_m=0.0, z_m=0.0),
            existence_probability=0.99,
        ),
        ObjectTrack(
            track_id="veh-adjacent",
            object_class=ObjectClass.VEHICLE,
            position=Point3D(x_m=adjacent_x, y_m=3.8, z_m=0.0),
            yaw_rad=0.0,
            extent_m=Point3D(x_m=4.1, y_m=1.9, z_m=1.6),
            velocity_mps=Point3D(x_m=0.3, y_m=0.0, z_m=0.0),
            existence_probability=0.96,
        ),
    ]
    scene = SceneSnapshotPayload(
        ego=EgoState(speed_mps=27.0, pose=Point3D(x_m=0.0, y_m=0.0, z_m=0.0), yaw_rad=0.0),
        tracks=tracks,
        lanes=[
            LanePolyline(lane_id="left", points=[Point3D(x_m=x, y_m=3.5, z_m=0.0) for x in range(-30, 121, 10)]),
            LanePolyline(lane_id="center", points=[Point3D(x_m=x, y_m=0.0, z_m=0.0) for x in range(-30, 121, 10)]),
            LanePolyline(lane_id="right", points=[Point3D(x_m=x, y_m=-3.5, z_m=0.0) for x in range(-30, 121, 10)]),
        ],
        ego_plan=EgoPlan(
            maneuver="lane_change_left" if 2.0 < simulation_time_s < 6.0 else "keep_lane",
            trajectory=[Point3D(x_m=i * 4.0, y_m=min(i * 0.35, 3.5), z_m=0.0) for i in range(1, 8)],
        ),
        provenance=Provenance.FUSION,
    )
    driver = DriverStatePayload(
        gaze_region=gaze_region if dms_available else GazeRegion.UNKNOWN,
        attention=attention if dms_available else AttentionState.UNKNOWN,
        workload=workload if dms_available else None,
        confidence=0.86 if dms_available else 0.0,
        available=dms_available,
        measurement_time_s=simulation_time_s if not stale_scene else max(simulation_time_s - 2.5, 0.0),
    )
    automation = AutomationStatePayload(mode="highway_assist", engaged=True, confidence=0.93)
    explanation = None
    if trigger_critical_event:
        explanation = ExplanationEventPayload(
            event_id=f"crit-lc-{int(simulation_time_s*10)}",
            title="Kritischer Hinweis",
            text="Abstand beim Spurwechsel knapp, Manöver wird gehalten.",
            referenced_track_id="veh-adjacent",
            critical=True,
        )
    elif trigger_noncritical_event:
        explanation = ExplanationEventPayload(
            event_id=f"info-lc-{int(simulation_time_s*10)}",
            title="Spurwechsel geplant",
            text="Geplanter Wechsel auf linke Spur wegen langsamem Vordermann.",
            referenced_track_id="veh-lead",
            critical=False,
        )
    return ScenarioStep(scene=scene, driver=driver, automation=automation, explanation=explanation)


def _data_quality_step(
    *,
    simulation_time_s: float,
    workload: float | None,
    dms_available: bool,
    attention: AttentionState,
    gaze_region: GazeRegion,
    trigger_critical_event: bool,
    trigger_noncritical_event: bool,
    stale_scene: bool,
) -> ScenarioStep:
    scene = SceneSnapshotPayload(
        ego=EgoState(speed_mps=13.0, pose=Point3D(x_m=0.0, y_m=0.0, z_m=0.0), yaw_rad=0.0),
        tracks=[],
        lanes=_base_lanes(),
        ego_plan=EgoPlan(maneuver="keep_lane", trajectory=[Point3D(x_m=i, y_m=0.0, z_m=0.0) for i in range(0, 20, 4)]),
        provenance=Provenance.SYNTHETIC,
    )
    dms_enabled = dms_available and not (2.0 <= simulation_time_s < 4.0)
    driver = DriverStatePayload(
        gaze_region=gaze_region if dms_enabled else GazeRegion.UNKNOWN,
        attention=attention if dms_enabled else AttentionState.UNKNOWN,
        workload=workload if dms_enabled else None,
        confidence=0.82 if dms_enabled else 0.0,
        available=dms_enabled,
        measurement_time_s=simulation_time_s if not stale_scene else max(simulation_time_s - 4.0, 0.0),
    )
    automation = AutomationStatePayload(mode="quality_monitor", engaged=False, confidence=0.7)
    explanation = None
    if trigger_critical_event:
        explanation = ExplanationEventPayload(
            event_id=f"dq-crit-{int(simulation_time_s*10)}",
            title="Kritischer Datenhinweis",
            text="Wichtige Sensordaten unterbrochen, nur Basisdarstellung aktiv.",
            critical=True,
        )
    elif trigger_noncritical_event or (4.0 <= simulation_time_s < 4.1):
        explanation = ExplanationEventPayload(
            event_id=f"dq-info-{int(simulation_time_s*10)}",
            title="DMS wieder verfügbar",
            text="Fahrerzustand wieder vorhanden; Policy wird aktualisiert.",
            critical=False,
        )
    return ScenarioStep(scene=scene, driver=driver, automation=automation, explanation=explanation)
