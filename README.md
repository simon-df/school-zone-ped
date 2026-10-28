# Adaptive In-Vehicle HMI Forschungsprototyp (Schema 0.1)

Dieses Repository enthält zusätzlich zum bestehenden `pedestrian_analysis`-Teil eine lauffähige erste Version einer portablen, adaptiven In-Vehicle-HMI-Forschungsplattform:

- **Backend:** FastAPI + Pydantic (`backend/`)
- **Frontend:** React + TypeScript (Strict) + Vite + React Three Fiber (`frontend/`)
- **Beispiele & Recordings:** `examples/`
- **Dokumentation:** `docs/`

> **Wichtig:** Die Mock-Daten sind deterministisch synthetisch. Es wird keine echte CARLA-/Unity-/Unreal-/Fahrzeuganbindung behauptet.

---

## Voraussetzungen

- Python 3.11+
- Node.js 20+ / npm 10+

---

## Installation

### Backend

```bash
cd /home/runner/work/school-zone-ped/school-zone-ped/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Frontend

```bash
cd /home/runner/work/school-zone-ped/school-zone-ped/frontend
npm install
```

---

## Start

### 1) Backend starten

```bash
cd /home/runner/work/school-zone-ped/school-zone-ped/backend
source .venv/bin/activate
python run_server.py
```

Backend läuft auf `http://127.0.0.1:8000`.

### 2) Frontend starten

```bash
cd /home/runner/work/school-zone-ped/school-zone-ped/frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Frontend läuft auf `http://127.0.0.1:5173`.

---

## URLs

- Workbench: `http://127.0.0.1:5173/workbench`
- Center Display: `http://127.0.0.1:5173/display/center`
- Panorama Display: `http://127.0.0.1:5173/display/panorama`
- HUD Display: `http://127.0.0.1:5173/display/hud`

Alle Flächen können unabhängig geöffnet werden und erhalten denselben autoritativen Zustand aus dem Backend.

---

## API (Mindestumfang)

- `GET  /api/v1/health`
- `GET  /api/v1/schema`
- `GET  /api/v1/capabilities`
- `POST /api/v1/ingest`
- `WS   /api/v1/stream`
- `PUT  /api/v1/preferences`
- `PUT  /api/v1/experiment`
- `POST /api/v1/interactions`
- `POST /api/v1/source`
- `POST /api/v1/mock/control`
- `POST /api/v1/recordings/start`
- `POST /api/v1/recordings/stop`
- `GET  /api/v1/recordings`
- `POST /api/v1/replay/control`

---

## Externer Publisher starten (ohne CARLA)

```bash
cd /home/runner/work/school-zone-ped/school-zone-ped
python examples/external_ingest.py --base-url http://127.0.0.1:8000 --hz 5
```

Vorher in der Workbench oder per API auf `source=external` umschalten.

---

## Mock, Recording, Replay bedienen

1. Workbench öffnen.
2. Quelle `Mock` wählen, Szenario auswählen (`city`, `highway`, `data_quality`).
3. Play/Pause/Reset, Geschwindigkeit, Workload, DMS/Attention, kritische/nichtkritische Events steuern.
4. Recording starten/stoppen.
5. Quelle auf `Replay` wechseln, Recording laden, Play/Pause/Seek/Reset nutzen.

Recordings werden lokal unter `examples/recordings/<recording_id>/` als `metadata.json` + `events.jsonl` abgelegt.

---

## Tests und Build

### Backend

```bash
cd /home/runner/work/school-zone-ped/school-zone-ped/backend
source .venv/bin/activate
pytest
```

### Frontend

```bash
cd /home/runner/work/school-zone-ped/school-zone-ped/frontend
npm run build
npm run test
npm run test:e2e
```

---

## Grenzen des Prototyps

- Keine echte Fahrzeugszene, kein Fahrsimulator im Projekt enthalten
- Keine sicherheitsvalidierte HMI-Freigabe für realen Fahrbetrieb
- Keine personenbezogenen Echt-DMS-Daten
- Kein ROS/Kafka/Redis/Cloud/Login integriert
- Replay seek rekonstruiert Zustand deterministisch aus aufgezeichneten Eingängen (kurze Sessions)
- Eine Workbench für mutierende Steuerung vorgesehen; keine komplexe Multi-User-Synchronisation

---

## Weitere Dokumentation

- Architektur: `docs/architecture.md`
- Nachrichten/Protokoll: `docs/protocol.md`
- Adaptionsregeln: `docs/adaptation-rules.md`
- Integrationshinweise: `docs/integration.md`
