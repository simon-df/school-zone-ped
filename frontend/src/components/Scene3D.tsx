import { Canvas } from '@react-three/fiber'
import type { ColorRepresentation } from 'three'

import { egoToThree } from '../perception/coordinates'
import type { SceneSnapshotPayload } from '../protocol/types'

interface Scene3DProps {
  scene: SceneSnapshotPayload | null
  highlightedTrackIds: string[]
  showPrediction: boolean
}

export function Scene3D({ scene, highlightedTrackIds, showPrediction }: Scene3DProps) {
  return (
    <div className="scene-canvas">
      <Canvas camera={{ position: [0, 14, 18], fov: 48 }}>
        <ambientLight intensity={0.9} />
        <pointLight position={[8, 16, 8]} intensity={0.8} />
        <group rotation={[-Math.PI / 2, 0, 0]}>
          <mesh position={[0, 0, 0]}>
            <planeGeometry args={[70, 50]} />
            <meshStandardMaterial color="#0e1722" />
          </mesh>

          {scene?.lanes.map((lane) => (
            <group key={lane.lane_id}>
              {lane.points.slice(0, lane.points.length - 1).map((point, index) => {
                const p0 = egoToThree(point)
                const p1 = egoToThree(lane.points[index + 1])
                const dx = p1.x - p0.x
                const dz = p1.z - p0.z
                const len = Math.hypot(dx, dz)
                return (
                  <mesh
                    key={`${lane.lane_id}-${index}`}
                    position={[(p0.x + p1.x) / 2, 0.02, (p0.z + p1.z) / 2]}
                    rotation={[0, Math.atan2(dx, dz), 0]}
                  >
                    <boxGeometry args={[0.12, 0.02, len]} />
                    <meshStandardMaterial color="#39516f" />
                  </mesh>
                )
              })}
            </group>
          ))}

          <mesh position={[0, 0.15, 0]}>
            <boxGeometry args={[1.8, 0.3, 3.8]} />
            <meshStandardMaterial color="#1aa4ff" />
          </mesh>

          {scene?.tracks.map((track) => {
            const p = egoToThree(track.position)
            const highlight = highlightedTrackIds.includes(track.track_id)
            return (
              <mesh key={track.track_id} position={[p.x, 0.18, p.z]}>
                <boxGeometry args={[track.extent_m.y_m, 0.28, track.extent_m.x_m]} />
                <meshStandardMaterial color={trackColor(track.object_class, highlight)} />
              </mesh>
            )
          })}

          {showPrediction &&
            scene?.tracks.flatMap((track) =>
              (track.predicted_trajectories ?? []).flatMap((prediction) =>
                prediction.points.map((point, index) => {
                  const p = egoToThree(point)
                  return (
                    <mesh key={`${track.track_id}-${prediction.hypothesis_id}-${index}`} position={[p.x, 0.1, p.z]}>
                      <sphereGeometry args={[0.15, 10, 10]} />
                      <meshStandardMaterial color="#f4b142" opacity={0.75} transparent />
                    </mesh>
                  )
                }),
              ),
            )}
        </group>
      </Canvas>
      <div className="scene-legend">
        <span>
          <b>Legende:</b> Größe der Boxen = Objektabmessung; orange Punkte = Vorhersage-Hypothesenpunkte;
          Existenzwahrscheinlichkeit wird im Objekt-Panel separat als Zahl angezeigt.
        </span>
      </div>
    </div>
  )
}

function trackColor(objectClass: string, highlighted: boolean): ColorRepresentation {
  if (highlighted) return '#ffdd57'
  switch (objectClass) {
    case 'pedestrian':
      return '#7dcfff'
    case 'vehicle':
      return '#7ccf66'
    case 'bicycle':
      return '#be95ff'
    default:
      return '#b8c0cc'
  }
}
