# Integration und Erweiterung

## Neue Datenquelle anbinden

1. Quelle auf Envelope-Format `0.1` abbilden.
2. Stabilen `source_id` und monoton steigende Sequenzen pro Nachrichtentyp liefern.
3. `POST /api/v1/source` mit `mode=external` aktivieren.
4. Nachrichten über `POST /api/v1/ingest` senden.
5. Optional Fähigkeiten über `GET /api/v1/capabilities` prüfen.

## Display-/Host-Integration

Aktuell implementiert:

- `BrowserHost` (`frontend/src/host/browserHost.ts`)

Geplante Integrationswege:

- separater Browser auf Labor-/Fahrzeugdisplay
- WebView/Offscreen-Browser als Textur in virtuellen Cockpit-Umgebungen
- alternativer nativer Renderer gegen denselben Daten-/Policy-Vertrag

## AR-Erweiterungsstelle

Eine spätere AR-Erweiterung kann semantische Direktiven wie `target_object_id` und `style` an den vorhandenen Vertrag anbinden. Diese Version implementiert bewusst keine AR- oder Engine-spezifischen Plugins.

## Sicherheits- und Betriebsgrenzen

- Prototyp für vertrauenswürdige lokale Umgebungen
- Backend standardmäßig auf Loopback (`127.0.0.1`)
- CORS nur für lokale Frontend-Origins
- Keine Produktionsfreigabe, keine Login-/Cloud-Abhängigkeit
