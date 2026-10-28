from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from app.protocol.models import InboundEnvelope


def test_protocol_fixture_parses() -> None:
    fixture_path = Path(__file__).resolve().parents[2] / "examples" / "protocol_fixture_scene_snapshot.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    adapter = TypeAdapter(InboundEnvelope)
    message = adapter.validate_python(payload)
    assert message.type == "scene.snapshot"
