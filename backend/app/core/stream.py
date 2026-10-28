from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from app.protocol.models import CanonicalState, StateSnapshotMessage, StateUpdateMessage


@dataclass
class ClientConnection:
    websocket: WebSocket
    queue: asyncio.Queue[dict[str, Any]] = field(default_factory=lambda: asyncio.Queue(maxsize=10))

    def __hash__(self) -> int:
        return id(self)


class StreamHub:
    def __init__(self) -> None:
        self._clients: set[ClientConnection] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, initial_state: CanonicalState) -> ClientConnection:
        await websocket.accept()
        client = ClientConnection(websocket=websocket)
        await websocket.send_text(
            StateSnapshotMessage(state=initial_state).model_dump_json()
        )
        async with self._lock:
            self._clients.add(client)
        return client

    async def disconnect(self, client: ClientConnection) -> None:
        async with self._lock:
            self._clients.discard(client)

    async def broadcast_state(self, state: CanonicalState) -> None:
        message = StateUpdateMessage(
            state=state,
            changed_at_simulation_time_s=state.last_simulation_time_s,
        ).model_dump()
        async with self._lock:
            clients = list(self._clients)
        for client in clients:
            await self._queue_latest_state(client, message)

    async def client_sender_loop(self, client: ClientConnection) -> None:
        try:
            while True:
                message = await client.queue.get()
                await client.websocket.send_text(json.dumps(message))
        except WebSocketDisconnect:
            await self.disconnect(client)
        except RuntimeError:
            await self.disconnect(client)

    async def _queue_latest_state(self, client: ClientConnection, message: dict[str, Any]) -> None:
        if client.queue.full():
            drained: list[dict[str, Any]] = []
            while not client.queue.empty():
                drained.append(client.queue.get_nowait())
            if drained:
                drained = [entry for entry in drained if entry.get("type") != "state.update"]
                for entry in drained[-8:]:
                    client.queue.put_nowait(entry)
        try:
            client.queue.put_nowait(message)
        except asyncio.QueueFull:
            # Drop oldest non-critical update and keep the latest state.
            _ = client.queue.get_nowait()
            client.queue.put_nowait(message)
