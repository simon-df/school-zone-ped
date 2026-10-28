import { useEffect, useMemo } from 'react'

import { StatusBar } from '../components/StatusBar'
import { BrowserHost } from '../host/browserHost'
import { useStateStream } from '../transport/ws'
import { CenterDisplay } from '../surfaces/CenterDisplay'
import { HudDisplay } from '../surfaces/HudDisplay'
import { PanoramaDisplay } from '../surfaces/PanoramaDisplay'
import { Workbench } from '../surfaces/Workbench'

export function AppRoot() {
  const host = useMemo(() => new BrowserHost(), [])
  const { state, connectionState } = useStateStream()

  useEffect(() => {
    host.reportReady()
    return () => host.dispose()
  }, [host])

  const configuration = host.getConfiguration()
  const route = configuration.route

  if (route.startsWith('/display/center')) {
    return (
      <div className="single-display">
        <StatusBar state={state} connectionState={connectionState} />
        <CenterDisplay displayId={configuration.displayId} state={state} />
      </div>
    )
  }
  if (route.startsWith('/display/panorama')) {
    return (
      <div className="single-display">
        <StatusBar state={state} connectionState={connectionState} />
        <PanoramaDisplay state={state} />
      </div>
    )
  }
  if (route.startsWith('/display/hud')) {
    return (
      <div className="single-display">
        <StatusBar state={state} connectionState={connectionState} />
        <HudDisplay state={state} />
      </div>
    )
  }

  return <Workbench displayId={configuration.displayId} state={state} connectionState={connectionState} />
}
