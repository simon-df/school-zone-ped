import { Scene3D } from '../components/Scene3D'
import type { CanonicalState } from '../protocol/types'
import { postInteraction } from '../transport/api'

interface CenterDisplayProps {
  displayId: string
  state: CanonicalState | null
}

export function CenterDisplay({ displayId, state }: CenterDisplayProps) {
  const tracks = state?.scene?.tracks ?? []
  const selected = state?.selected_track_id
  const selectedTrack = tracks.find((track) => track.track_id === selected)

  return (
    <section className="surface center-surface">
      <div className="surface-header">
        <h2>Center Display</h2>
        <p>Abstrahierte Fusion-Szene mit Vorhersage und Plan.</p>
      </div>

      <Scene3D
        scene={state?.scene ?? null}
        highlightedTrackIds={state?.policy?.highlighted_track_ids ?? []}
        showPrediction={state?.policy?.show_prediction ?? false}
      />

      <div className="center-grid">
        <div className="card">
          <h3>Objekte</h3>
          <div className="scroll">
            {tracks.map((track) => (
              <button
                key={track.track_id}
                className={track.track_id === selected ? 'row-button active' : 'row-button'}
                onClick={() => postInteraction(displayId, 'select_object', track.track_id)}
              >
                <span>{track.track_id}</span>
                <small>{track.object_class}</small>
              </button>
            ))}
            {tracks.length === 0 && <p>Keine aktiven Tracks.</p>}
          </div>
          <button className="secondary" onClick={() => postInteraction(displayId, 'clear_selection')}>
            Auswahl aufheben
          </button>
        </div>

        <div className="card">
          <h3>Track-Details</h3>
          {selectedTrack ? (
            <dl>
              <dt>ID</dt>
              <dd>{selectedTrack.track_id}</dd>
              <dt>Existenz</dt>
              <dd>{selectedTrack.existence_probability?.toFixed(2) ?? 'n/a'}</dd>
              <dt>Position [x,y]</dt>
              <dd>
                {selectedTrack.position.x_m.toFixed(2)} m, {selectedTrack.position.y_m.toFixed(2)} m
              </dd>
              <dt>Geschwindigkeit [x,y]</dt>
              <dd>
                {selectedTrack.velocity_mps.x_m.toFixed(2)} m/s, {selectedTrack.velocity_mps.y_m.toFixed(2)} m/s
              </dd>
            </dl>
          ) : (
            <p>Objekt im linken Panel auswählen.</p>
          )}
        </div>

        <div className="card">
          <h3>Erklärungen</h3>
          <div className="scroll">
            {(state?.active_explanations ?? []).map((event) => (
              <article key={event.event_id} className={event.critical ? 'explanation critical' : 'explanation'}>
                <h4>{event.title}</h4>
                <p>{event.text}</p>
                <small>Objekt: {event.referenced_track_id ?? '—'}</small>
              </article>
            ))}
            {(state?.active_explanations.length ?? 0) === 0 && <p>Keine aktiven Erklärungsereignisse.</p>}
          </div>
          <div className="button-row">
            <button onClick={() => postInteraction(displayId, 'expand_explanation')}>Details erweitern</button>
            <button className="secondary" onClick={() => postInteraction(displayId, 'reduce_explanation')}>
              Details reduzieren
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}
