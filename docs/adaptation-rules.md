# Regelmatrix (initiale experimentelle Regeln)

## Prioritäten

1. **Kritische Ereignisse sichtbar halten** (`CRITICAL_EVENT`)
2. **Hohe valide Belastung begrenzt Dichte/Details** (`HIGH_WORKLOAD`)
3. **Unbekannter Fahrerzustand ⇒ konservative Basisdarstellung** (`DRIVER_STATE_UNKNOWN`)
4. Profil- und manuelle Detailwünsche nur innerhalb dieser Grenzen
5. Veraltete Szenendaten markieren und datenabhängige Details reduzieren (`SCENE_DATA_STALE`)

## Schwellen und Zeitverhalten

| Regel | Wert |
|---|---|
| High-Workload-Schwelle | `>= 0.7` |
| Rückkehrschwelle | `<= 0.5` |
| Entprellung (stabil) | `1.0 s` |
| Mindestverweildauer Dichtestufe | `3.0 s` |

## Bedingungsspezifisches Verhalten

- **`static`**: Dichte bleibt `medium` (`STATIC_BASELINE`), unabhängig von Workload.
- **`adaptive`**: Dichteanpassung nach Workload/Verfügbarkeit inkl. Entprellung + Mindestverweildauer.

## Profilwirkung (experimentelle Presets)

- `safety`: konservative Standarddetails
- `efficiency`: Standard
- `control`: leichter zugängliche ausführlichere Erklärungen
- `tech`: zusätzliche Prediction/Unsicherheitsdetails bei niedriger valider Belastung

## Reason Codes

Verwendete Codes:

- `STATIC_BASELINE`
- `HIGH_WORKLOAD`
- `DRIVER_STATE_UNKNOWN`
- `TECH_PROFILE_DETAIL`
- `CRITICAL_EVENT`
- `SCENE_DATA_STALE`
- `MANUAL_DETAIL_REQUEST`
