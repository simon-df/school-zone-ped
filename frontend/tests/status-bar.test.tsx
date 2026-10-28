import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { StatusBar } from '../src/components/StatusBar'
import type { CanonicalState } from '../src/protocol/types'

const state: CanonicalState = {
  schema_version: '0.1',
  session_id: 's1',
  source_mode: 'mock',
  condition: 'adaptive',
  preferences: { profile: 'safety', explanation_detail_level: 'medium', show_prediction_preference: true },
  scene: null,
  driver_state: null,
  automation_state: null,
  active_explanations: [],
  selected_track_id: null,
  manual_detail_requested: false,
  policy: null,
  system_status: {
    source_mode: 'mock',
    source_id: 'mock',
    data_connection: 'disconnected',
    scene_data_stale: true,
    scene_age_s: 2.2,
    replay_paused: false,
  },
  last_simulation_time_s: 2.2,
  source_capabilities: {
    source_id: 'mock',
    supports_objects: true,
    supports_lanes: true,
    supports_prediction: true,
    supports_gaze: true,
  },
}

describe('StatusBar', () => {
  it('shows stale data and disconnection state', () => {
    render(<StatusBar state={state} connectionState="reconnecting" />)
    expect(screen.getByText('Datenstatus: veraltet')).toBeInTheDocument()
    expect(screen.getByText('Verbindungsstatus: disconnected')).toBeInTheDocument()
  })
})
