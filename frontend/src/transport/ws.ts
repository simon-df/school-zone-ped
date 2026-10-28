import { useEffect, useRef, useState } from 'react'

import type { CanonicalState, StateSnapshotMessage, StateUpdateMessage } from '../protocol/types'

type IncomingMessage = StateSnapshotMessage | StateUpdateMessage

export function useStateStream() {
  const [state, setState] = useState<CanonicalState | null>(null)
  const [connectionState, setConnectionState] = useState<'connecting' | 'connected' | 'reconnecting' | 'disconnected'>(
    'connecting',
  )
  const retryRef = useRef(0)
  const closedRef = useRef(false)

  useEffect(() => {
    closedRef.current = false
    let socket: WebSocket | null = null
    let retryTimer: number | null = null

    const connect = () => {
      setConnectionState(retryRef.current === 0 ? 'connecting' : 'reconnecting')
      socket = new WebSocket(buildWsUrl('/api/v1/stream'))
      socket.onopen = () => {
        retryRef.current = 0
        setConnectionState('connected')
      }
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data) as IncomingMessage
        if (message.type === 'state.snapshot' || message.type === 'state.update') {
          setState(message.state)
        }
      }
      socket.onclose = () => {
        if (closedRef.current) {
          setConnectionState('disconnected')
          return
        }
        retryRef.current += 1
        const backoffMs = Math.min(1000 * 2 ** retryRef.current, 8000)
        retryTimer = window.setTimeout(connect, backoffMs)
      }
      socket.onerror = () => {
        socket?.close()
      }
    }

    connect()

    return () => {
      closedRef.current = true
      if (retryTimer !== null) window.clearTimeout(retryTimer)
      socket?.close()
    }
  }, [])

  return { state, connectionState }
}

function buildWsUrl(path: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${protocol}://${window.location.host}${path}`
}
