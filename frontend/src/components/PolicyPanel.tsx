import type { CanonicalState } from '../protocol/types'

export function PolicyPanel({ state }: { state: CanonicalState | null }) {
  if (!state?.policy) return <div className="card">Warte auf Policy...</div>
  const { policy } = state
  return (
    <div className="card">
      <h3>Policy</h3>
      <div className="policy-grid">
        <span>Dichte</span>
        <strong>{policy.information_density}</strong>
        <span>Prediction</span>
        <strong>{policy.show_prediction ? 'an' : 'aus'}</strong>
        <span>Unsicherheit</span>
        <strong>{policy.show_uncertainty ? 'an' : 'aus'}</strong>
        <span>Regelversion</span>
        <strong>{policy.rule_version}</strong>
      </div>
      <h4>Reason Codes</h4>
      <ul>
        {policy.reason_codes.map((reasonCode) => (
          <li key={reasonCode}>{reasonCode}</li>
        ))}
      </ul>
    </div>
  )
}
