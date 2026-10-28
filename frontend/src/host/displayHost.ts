export interface DisplayConfiguration {
  displayId: string
  route: string
  aspectRatio?: number
}

export interface DisplayEvent {
  displayId: string
  eventType: string
  detail: Record<string, unknown>
}

export interface DisplayHost {
  getConfiguration(): DisplayConfiguration
  reportReady(): void
  reportDisplayEvent(event: DisplayEvent): void
  dispose(): void
}
