import type { Point3D } from '../protocol/types'

export interface ThreePoint {
  x: number
  y: number
  z: number
}

export function egoToThree(point: Point3D): ThreePoint {
  return {
    x: normalizeZero(-point.y_m),
    y: normalizeZero(point.z_m),
    z: normalizeZero(-point.x_m),
  }
}

function normalizeZero(value: number): number {
  return Object.is(value, -0) ? 0 : value
}
