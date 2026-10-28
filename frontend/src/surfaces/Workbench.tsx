import { useEffect, useMemo, useState } from 'react'

import { PolicyPanel } from '../components/PolicyPanel'
import { StatusBar } from '../components/StatusBar'
import type { CanonicalState, PreferenceProfile } from '../protocol/types'
import {
  controlMock,
  controlReplay,
  listRecordings,
  postInteraction,
  setExperiment,
  setPreferences,
  setSource,
  startRecording,
  stopRecording,
} from '../transport/api'
import { CenterDisplay } from './CenterDisplay'
import { HudDisplay } from './HudDisplay'
import { PanoramaDisplay } from './PanoramaDisplay'

interface WorkbenchProps {
  displayId: string
  state: CanonicalState | null
  connectionState: 'connecting' | 'connected' | 'reconnecting' | 'disconnected'
}

export function Workbench({ displayId, state, connectionState }: WorkbenchProps) {
  const [recordingId, setRecordingId] = useState<string | null>(null)
  const [recordings, setRecordings] = useState<Array<Record<string, unknown>>>([])
  const [selectedRecording, setSelectedRecording] = useState<string>('')

  useEffect(() => {
    listRecordings().then((response) => setRecordings(response.recordings)).catch(() => undefined)
  }, [])

  const isMock = state?.source_mode === 'mock'
  const isReplay = state?.source_mode === 'replay'
  const profile = state?.preferences.profile ?? 'safety'

  const activeReasons = useMemo(() => state?.policy?.reason_codes.join(', ') ?? '—', [state?.policy?.reason_codes])

  return (
    <main className="workbench">
      <StatusBar state={state} connectionState={connectionState} />
      <header>
        <h1>Adaptive Automotive HMI Research Workbench</h1>
        <p>Alle dargestellten Sensordaten in diesem Prototyp sind synthetisch, sofern nicht External-Ingest aktiv ist.</p>
      </header>

      <div className="workbench-grid">
        <section className="card controls">
          <h3>Quelle & Szenario</h3>
          <div className="button-row">
            <button onClick={() => setSource('mock')}>Mock</button>
            <button disabled={!selectedRecording} onClick={() => selectedRecording && setSource('replay', selectedRecording)}>
              Replay
            </button>
            <button onClick={() => setSource('external')}>External</button>
          </div>
          <div className="hint">
            Mock-Regler gelten nur im Modus Mock und überschreiben im External-Modus keine Eingangsdaten.
          </div>
          <label>
            Szenario
            <select
              disabled={!isMock}
              onChange={(event) => controlMock({ scenario_id: event.target.value })}
              defaultValue="city"
            >
              <option value="city">Stadtverkehr</option>
              <option value="highway">Autobahn</option>
              <option value="data_quality">Datenqualität</option>
            </select>
          </label>
          <div className="button-row">
            <button disabled={!isMock} onClick={() => controlMock({ running: true })}>
              Play
            </button>
            <button disabled={!isMock} className="secondary" onClick={() => controlMock({ running: false })}>
              Pause
            </button>
            <button disabled={!isMock} className="secondary" onClick={() => controlMock({ action: 'reset' })}>
              Reset
            </button>
          </div>
          <label>
            Geschwindigkeit
            <input
              disabled={!isMock}
              type="range"
              min={0.2}
              max={3}
              step={0.1}
              defaultValue={1}
              onChange={(event) => controlMock({ speed: Number(event.target.value) })}
            />
          </label>
          <label>
            Timeline
            <input
              disabled={!isReplay}
              type="range"
              min={0}
              max={120}
              step={0.5}
              value={Math.min(state?.last_simulation_time_s ?? 0, 120)}
              onChange={(event) => controlReplay({ action: 'seek', seek_time_s: Number(event.target.value) })}
            />
          </label>
          <div className="button-row">
            <button disabled={!isReplay} onClick={() => controlReplay({ action: 'play' })}>
              Replay Play
            </button>
            <button disabled={!isReplay} className="secondary" onClick={() => controlReplay({ action: 'pause' })}>
              Replay Pause
            </button>
          </div>
        </section>

        <section className="card controls">
          <h3>Versuch & Profil</h3>
          <div className="button-row">
            <button onClick={() => setExperiment('static')}>Static</button>
            <button onClick={() => setExperiment('adaptive')}>Adaptive</button>
          </div>
          <label>
            Profil
            <select value={profile} onChange={(event) => setPreferences(event.target.value as PreferenceProfile)}>
              <option value="safety">Sicherheit</option>
              <option value="efficiency">Effizienz</option>
              <option value="control">Kontrolle</option>
              <option value="tech">Technik</option>
            </select>
          </label>
          <div className="button-row">
            <button onClick={() => postInteraction(displayId, 'set_manual_detail_request', true)}>
              Manueller Detailwunsch
            </button>
            <button className="secondary" onClick={() => postInteraction(displayId, 'set_manual_detail_request', false)}>
              Detailwunsch zurücknehmen
            </button>
          </div>
          <label>
            Mock-Belastung
            <input
              disabled={!isMock}
              type="range"
              min={0}
              max={1}
              step={0.01}
              defaultValue={0.35}
              onChange={(event) => controlMock({ workload: Number(event.target.value) })}
            />
          </label>
          <div className="button-row">
            <button disabled={!isMock} onClick={() => controlMock({ dms_available: false })}>
              DMS aus
            </button>
            <button disabled={!isMock} className="secondary" onClick={() => controlMock({ dms_available: true })}>
              DMS an
            </button>
          </div>
          <div className="button-row">
            <button disabled={!isMock} onClick={() => controlMock({ attention: 'distracted' })}>
              Aufmerksamkeit: abgelenkt
            </button>
            <button disabled={!isMock} className="secondary" onClick={() => controlMock({ attention: 'attentive' })}>
              aufmerksam
            </button>
          </div>
        </section>

        <section className="card controls">
          <h3>Ereignisse & Datenqualität</h3>
          <div className="button-row">
            <button disabled={!isMock} onClick={() => controlMock({ trigger_critical_event: true })}>
              Kritisches Mock-Ereignis
            </button>
            <button disabled={!isMock} className="secondary" onClick={() => controlMock({ trigger_noncritical_event: true })}>
              Nichtkritisches Mock-Ereignis
            </button>
          </div>
          <div className="button-row">
            <button disabled={!isMock} onClick={() => controlMock({ stale_scene: true })}>
              Daten veralten lassen
            </button>
            <button disabled={!isMock} className="secondary" onClick={() => controlMock({ stale_scene: false })}>
              Datenalter normal
            </button>
          </div>
          <div className="button-row">
            <button disabled={!isMock} onClick={() => controlMock({ disconnected: true })}>
              Verbindungsausfall
            </button>
            <button disabled={!isMock} className="secondary" onClick={() => controlMock({ disconnected: false })}>
              Verbindung wiederherstellen
            </button>
          </div>
          <p>Aktive Reason Codes: {activeReasons}</p>
          <p>Policy-ID: {state?.policy?.policy_id ?? '—'}</p>
        </section>

        <section className="card controls">
          <h3>Recording / Replay</h3>
          <div className="button-row">
            <button
              onClick={async () => {
                const response = await startRecording()
                setRecordingId(response.recording_id)
                const listed = await listRecordings()
                setRecordings(listed.recordings)
              }}
            >
              Recording starten
            </button>
            <button
              className="secondary"
              onClick={async () => {
                await stopRecording()
                setRecordingId(null)
                const listed = await listRecordings()
                setRecordings(listed.recordings)
              }}
            >
              Recording stoppen
            </button>
          </div>
          <p>Aktives Recording: {recordingId ?? 'kein aktives Recording'}</p>
          <label>
            Aufzeichnung auswählen
            <select value={selectedRecording} onChange={(event) => setSelectedRecording(event.target.value)}>
              <option value="">—</option>
              {recordings.map((entry) => (
                <option key={String(entry.recording_id)} value={String(entry.recording_id)}>
                  {String(entry.recording_id)}
                </option>
              ))}
            </select>
          </label>
          <div className="button-row">
            <button
              disabled={!selectedRecording}
              onClick={async () => {
                await setSource('replay', selectedRecording)
                await controlReplay({ action: 'load', recording_id: selectedRecording })
              }}
            >
              Replay laden
            </button>
            <button disabled={!selectedRecording} className="secondary" onClick={() => controlReplay({ action: 'reset' })}>
              Replay reset
            </button>
          </div>
        </section>
      </div>

      <PolicyPanel state={state} />
      <CenterDisplay displayId={displayId} state={state} />
      <PanoramaDisplay state={state} />
      <HudDisplay state={state} />
    </main>
  )
}
