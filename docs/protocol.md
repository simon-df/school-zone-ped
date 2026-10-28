# Protokoll und Nachrichtenvertrag (`schema_version: 0.1`)

## Envelope

Alle Ingest-Nachrichten verwenden:

```text
schema_version
type
session_id
source_id
sequence
simulation_time_s
payload
```

`received_monotonic_s` wird serverseitig beim Recording ergänzt.

## Typen

- `scene.snapshot` (vollständiger Szenenersatz, kein Delta)
- `driver.state`
- `automation.state`
- `explanation.event`
- `user.preferences`
- `presentation.policy` (serverseitig autoritativ)
- `interaction.event`
- `system.status`

## WebSocket-Vertrag

- Endpoint: `WS /api/v1/stream`
- Initiale Nachricht: `state.snapshot` (vollständiger Zustand)
- Folge: `state.update` bei Änderungen
- Langsame Clients erhalten den neuesten Zustand bevorzugt (bounded queue / latest-state Verhalten)

## Sequenz- und Versionsregeln

- Nur `schema_version = "0.1"` wird akzeptiert.
- Sequenzen werden pro `(source_id, type, session_id)` überwacht.
- Doppelte oder veraltete Sequenzen werden mit Fehler abgelehnt.
- Bei neuer Session-ID darf Sequenz bei 0/1 neu starten.

## Beispiel

Siehe `examples/protocol_fixture_scene_snapshot.json` für ein vollständiges `scene.snapshot`-Fixture.
