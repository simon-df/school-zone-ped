"""Simple external ingest publisher without CARLA dependency."""

from __future__ import annotations

import argparse
import itertools
import time

import httpx


def build_scene(sequence: int, simulation_time_s: float) -> dict:
    track_x = 18.0 - simulation_time_s * 1.1
    return {
        "schema_version": "0.1",
        "type": "scene.snapshot",
        "session_id": "external-demo-session",
        "source_id": "external-demo",
        "sequence": sequence,
        "simulation_time_s": simulation_time_s,
        "payload": {
            "ego": {
                "speed_mps": 6.0,
                "pose": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0},
                "yaw_rad": 0.0,
            },
            "tracks": [
                {
                    "track_id": "ext-track-1",
                    "object_class": "vehicle",
                    "position": {"x_m": track_x, "y_m": -1.2, "z_m": 0.0},
                    "yaw_rad": 0.0,
                    "extent_m": {"x_m": 4.2, "y_m": 1.8, "z_m": 1.6},
                    "velocity_mps": {"x_m": -1.1, "y_m": 0.0, "z_m": 0.0},
                    "existence_probability": 0.98,
                }
            ],
            "lanes": [
                {
                    "lane_id": "lane-main",
                    "points": [
                        {"x_m": -10.0, "y_m": 0.0, "z_m": 0.0},
                        {"x_m": 50.0, "y_m": 0.0, "z_m": 0.0},
                    ],
                }
            ],
            "ego_plan": {
                "maneuver": "keep_lane",
                "trajectory": [
                    {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0},
                    {"x_m": 20.0, "y_m": 0.0, "z_m": 0.0},
                ],
            },
            "provenance": "perception",
        },
    }


def build_driver(sequence: int, simulation_time_s: float) -> dict:
    return {
        "schema_version": "0.1",
        "type": "driver.state",
        "session_id": "external-demo-session",
        "source_id": "external-demo",
        "sequence": sequence,
        "simulation_time_s": simulation_time_s,
        "payload": {
            "gaze_region": "road",
            "attention": "attentive",
            "workload": 0.42,
            "confidence": 0.85,
            "available": True,
            "measurement_time_s": simulation_time_s,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--hz", type=float, default=5.0)
    args = parser.parse_args()

    period_s = 1.0 / args.hz
    client = httpx.Client(base_url=args.base_url, timeout=5.0)
    client.post("/api/v1/source", json={"mode": "external"}).raise_for_status()
    print("Publishing external ingest events to /api/v1/ingest ...")

    for tick in itertools.count(start=1):
        t = tick * period_s
        scene = build_scene(sequence=tick, simulation_time_s=t)
        driver = build_driver(sequence=tick, simulation_time_s=t)
        client.post("/api/v1/ingest", json=scene).raise_for_status()
        client.post("/api/v1/ingest", json=driver).raise_for_status()
        time.sleep(period_s)


if __name__ == "__main__":
    main()
