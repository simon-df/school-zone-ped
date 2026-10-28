import type { CanonicalState } from '../protocol/types'

export function HudDisplay({ state }: { state: CanonicalState | null }) {
  return (
    <section className="surface hud-surface">
      <h2>HUD (2D reduziert)</h2>
      <div className="hud-strip">
        <div>
          <span>Ego</span>
          <strong>{state?.scene?.ego.speed_mps.toFixed(1) ?? '0.0'} m/s</strong>
        </div>
        <div>
          <span>Automation</span>
          <strong>{state?.automation_state?.engaged ? 'ON' : 'OFF'}</strong>
        </div>
        <div>
          <span>Dichte</span>
          <strong>{state?.policy?.information_density ?? '-'}</strong>
        </div>
        <div>
          <span>Hinweise</span>
          <strong>{state?.active_explanations.filter((entry) => entry.critical).length ?? 0}</strong>
        </div>
      </div>
      <p className="hint">
        Hinweis: Diese Fläche ist ein Forschungs-Mock und kein optisch kalibriertes, räumlich registriertes AR-HUD.
      </p>
    </section>
  )
}
