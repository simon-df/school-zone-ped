import { describe, expect, it } from 'vitest'

import { egoToThree } from '../src/perception/coordinates'

describe('egoToThree', () => {
  it('maps forward x to negative z for top-down camera', () => {
    const converted = egoToThree({ x_m: 4, y_m: 0, z_m: 0 })
    expect(converted).toEqual({ x: 0, y: 0, z: -4 })
  })

  it('maps left-positive y to negative x', () => {
    const converted = egoToThree({ x_m: 0, y_m: 3, z_m: 0 })
    expect(converted).toEqual({ x: -3, y: 0, z: 0 })
  })

  it('keeps up-positive z on three y axis', () => {
    const converted = egoToThree({ x_m: 0, y_m: 0, z_m: 2 })
    expect(converted).toEqual({ x: 0, y: 2, z: 0 })
  })
})
