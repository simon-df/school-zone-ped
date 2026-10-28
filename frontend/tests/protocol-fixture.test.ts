import { describe, expect, it } from 'vitest'

import type { SceneSnapshotPayload } from '../src/protocol/types'

describe('protocol fixture typing', () => {
  it('matches frontend scene snapshot contract', () => {
    const fixture: SceneSnapshotPayload = {
      ego: {
        speed_mps: 4,
        pose: { x_m: 0, y_m: 0, z_m: 0 },
        yaw_rad: 0,
      },
      tracks: [
        {
          track_id: 'obj-1',
          object_class: 'vehicle',
          position: { x_m: 10, y_m: -1, z_m: 0 },
          yaw_rad: 0,
          extent_m: { x_m: 4.2, y_m: 1.8, z_m: 1.6 },
          velocity_mps: { x_m: -0.5, y_m: 0, z_m: 0 },
          existence_probability: 0.9,
        },
      ],
      lanes: [{ lane_id: 'lane-1', points: [{ x_m: 0, y_m: 0, z_m: 0 }] }],
      ego_plan: { maneuver: 'keep_lane', trajectory: [{ x_m: 5, y_m: 0, z_m: 0 }] },
      provenance: 'fusion',
    }
    expect(fixture.tracks[0].track_id).toBe('obj-1')
  })
})
