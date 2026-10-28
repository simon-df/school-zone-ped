import { expect, test } from '@playwright/test'

test('workbench page renders', async ({ page }) => {
  await page.goto('/workbench')
  await expect(page.getByRole('heading', { name: /Adaptive Automotive HMI Research Workbench/ })).toBeVisible()
})
