from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


@dataclass
class RecordingSession:
    recording_id: str
    metadata_path: Path
    events_path: Path
    started_at: str
    active: bool = True


class RecordingManager:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._active: RecordingSession | None = None

    async def start(self, state_snapshot: dict) -> RecordingSession:
        async with self._lock:
            if self._active is not None and self._active.active:
                return self._active
            recording_id = f"rec-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
            folder = self._base_dir / recording_id
            folder.mkdir(parents=True, exist_ok=True)
            metadata_path = folder / "metadata.json"
            events_path = folder / "events.jsonl"
            started_at = datetime.now(UTC).isoformat()
            metadata_path.write_text(
                json.dumps(
                    {
                        "recording_id": recording_id,
                        "started_at_utc": started_at,
                        "state_at_start": state_snapshot,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            events_path.touch()
            self._active = RecordingSession(
                recording_id=recording_id,
                metadata_path=metadata_path,
                events_path=events_path,
                started_at=started_at,
            )
            return self._active

    async def stop(self) -> RecordingSession | None:
        async with self._lock:
            if self._active is None:
                return None
            active = self._active
            active.active = False
            metadata = json.loads(active.metadata_path.read_text(encoding="utf-8"))
            metadata["stopped_at_utc"] = datetime.now(UTC).isoformat()
            active.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            self._active = None
            return active

    async def record_event(self, event_type: str, payload: dict) -> None:
        async with self._lock:
            if self._active is None or not self._active.active:
                return
            line = json.dumps(
                {
                    "event_type": event_type,
                    "payload": payload,
                    "recorded_at_utc": datetime.now(UTC).isoformat(),
                }
            )
            with self._active.events_path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")

    async def list_recordings(self) -> list[dict]:
        recordings: list[dict] = []
        for folder in sorted(self._base_dir.glob("rec-*"), reverse=True):
            metadata_path = folder / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["active"] = (
                self._active is not None
                and self._active.recording_id == metadata.get("recording_id")
                and self._active.active
            )
            recordings.append(metadata)
        return recordings

    async def read_events(self, recording_id: str) -> list[dict]:
        events_path = self._base_dir / recording_id / "events.jsonl"
        if not events_path.exists():
            raise FileNotFoundError(f"recording not found: {recording_id}")
        events: list[dict] = []
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        return events
