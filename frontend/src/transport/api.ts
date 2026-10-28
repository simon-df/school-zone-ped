import type { PreferenceProfile } from '../protocol/types'

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!response.ok) {
    throw new Error(`${response.status}: ${await response.text()}`)
  }
  return (await response.json()) as T
}

export async function setSource(mode: 'mock' | 'replay' | 'external', recordingId?: string) {
  return requestJson('/api/v1/source', {
    method: 'POST',
    body: JSON.stringify({ mode, recording_id: recordingId }),
  })
}

export async function setExperiment(condition: 'static' | 'adaptive') {
  return requestJson('/api/v1/experiment', {
    method: 'PUT',
    body: JSON.stringify({ condition }),
  })
}

export async function setPreferences(profile: PreferenceProfile) {
  return requestJson('/api/v1/preferences', {
    method: 'PUT',
    body: JSON.stringify({
      profile,
      explanation_detail_level: 'medium',
      show_prediction_preference: true,
    }),
  })
}

export async function postInteraction(displayId: string, action: string, value?: string | boolean) {
  return requestJson('/api/v1/interactions', {
    method: 'POST',
    body: JSON.stringify({ display_id: displayId, action, value }),
  })
}

export async function controlMock(payload: Record<string, unknown>) {
  return requestJson('/api/v1/mock/control', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function startRecording() {
  return requestJson<{ recording_id: string }>('/api/v1/recordings/start', { method: 'POST' })
}

export async function stopRecording() {
  return requestJson<{ recording_id: string | null }>('/api/v1/recordings/stop', { method: 'POST' })
}

export async function listRecordings() {
  return requestJson<{ recordings: Array<Record<string, unknown>> }>('/api/v1/recordings')
}

export async function controlReplay(payload: Record<string, unknown>) {
  return requestJson('/api/v1/replay/control', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}
