import type { DisplayConfiguration, DisplayEvent, DisplayHost } from './displayHost'

export class BrowserHost implements DisplayHost {
  private readonly config: DisplayConfiguration

  public constructor() {
    this.config = {
      displayId: this.resolveDisplayId(),
      route: window.location.pathname,
    }
  }

  public getConfiguration(): DisplayConfiguration {
    return this.config
  }

  public reportReady(): void {
    // no-op in browser host
  }

  public reportDisplayEvent(event: DisplayEvent): void {
    window.dispatchEvent(new CustomEvent('display-event', { detail: event }))
  }

  public dispose(): void {
    // no-op in browser host
  }

  private resolveDisplayId(): string {
    if (window.location.pathname.includes('/display/center')) return 'center'
    if (window.location.pathname.includes('/display/panorama')) return 'panorama'
    if (window.location.pathname.includes('/display/hud')) return 'hud'
    return 'workbench'
  }
}
