import type { CanonicalState } from '../protocol/types'

export function PanoramaDisplay({ state }: { state: CanonicalState | null }) {
  return (
    <section className="surface panorama-surface">
      <h2>Panorama Display</h2>
      <div className="panorama-grid">
        <div className="card">
          <h3>Automationsstatus</h3>
          <p>Modus: {state?.automation_state?.mode ?? 'n/a'}</p>
          <p>Eingriff: {state?.automation_state?.engaged ? 'aktiv' : 'inaktiv'}</p>
          <p>Konfidenz: {state?.automation_state?.confidence?.toFixed(2) ?? 'n/a'}</p>
        </div>
        <div className="card">
          <h3>Fahrerzustand</h3>
          <p>Blick: {state?.driver_state?.gaze_region ?? 'unknown'}</p>
          <p>Aufmerksamkeit: {state?.driver_state?.attention ?? 'unknown'}</p>
          <p>Belastung: {state?.driver_state?.workload?.toFixed(2) ?? 'unknown'}</p>
        </div>
        <div className="card">
          <h3>Persistenter Status</h3>
          <p>Quelle: {state?.source_mode ?? '-'}</p>
          <p>Datenalter: {state?.system_status.scene_age_s.toFixed(2) ?? '0.00'}s</p>
          <p>Policy-Dichte: {state?.policy?.information_density ?? '-'}</p>
        </div>
      </div>
    </section>
  )
}
