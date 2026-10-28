import type { CanonicalState } from '../protocol/types'

interface StatusBarProps {
  state: CanonicalState | null
  connectionState: 'connecting' | 'connected' | 'reconnecting' | 'disconnected'
}

export function StatusBar({ state, connectionState }: StatusBarProps) {
  return (
    <div className="status-bar">
      <span>WS: {connectionState}</span>
      <span>Quelle: {state?.source_mode ?? '-'}</span>
      <span>Provenienz: {state?.scene?.provenance ?? '-'}</span>
      <span>Datenstatus: {state?.system_status.scene_data_stale ? 'veraltet' : 'aktuell'}</span>
      <span>Verbindungsstatus: {state?.system_status.data_connection ?? '-'}</span>
      <span>Zeit: {state?.last_simulation_time_s.toFixed(2) ?? '0.00'}s</span>
    </div>
  )
}
