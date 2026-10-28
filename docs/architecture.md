# Architekturübersicht: Adaptive Automotive HMI Prototype

## Komponenten

- **Backend (FastAPI, `backend/app`)**
  - `protocol/`: versionierter Datenvertrag (Pydantic, Schema `0.1`)
  - `core/`: kanonischer Zustand, Adaptions-Core, Szenariogenerator, Stream-Hub
  - `sources/`: Mock-Quelle und Replay-Quelle
  - `recording/`: JSONL-Aufzeichnung + Metadaten
  - `api/`: HTTP-/WS-Endpunkte
- **Frontend (React + TypeScript + Vite, `frontend/src`)**
  - `transport/`: HTTP + WebSocket mit Reconnect-Backoff
  - `host/`: `DisplayHost`-Abstraktion (`BrowserHost`)
  - `surfaces/`: Workbench, Center, Panorama, HUD
  - `perception/`: Koordinatentransformation Ego → Three.js
  - `components/`: Rendering- und Diagnosekomponenten
- **Examples (`examples/`)**
  - `external_ingest.py`: ausführbarer Ingest-Publisher ohne Simulatorabhängigkeit
  - `recordings/`: lokale Recording-Artefakte

## Datenfluss

1. Quelle (`mock`, `replay`, `external`) erzeugt Envelopes.
2. Backend validiert/normalisiert, prüft Sequenzen und aktualisiert den kanonischen Zustand.
3. Adaptions-Core berechnet zentral die `presentation.policy`.
4. Zustand wird an alle Display-Clients über `/api/v1/stream` verteilt.
5. Frontend rendert ausschließlich aus dem kanonischen Zustand.
6. Interaktionen gehen als Ereignisse zurück ans Backend (`/api/v1/interactions`).
7. Recording schreibt Eingänge, Konfiguration, Interaktionen und Policy-Kontext als JSONL.

## Verantwortlichkeitstrennung

- Datenquellen sind von Adaptionsregeln getrennt.
- Adaptions-Core ist backend-zentral und deterministisch testbar.
- Frontend enthält keine fahrzeugsimulator-spezifischen Objekte.
- Host/Transport sind von UI-Komponenten gekapselt.
