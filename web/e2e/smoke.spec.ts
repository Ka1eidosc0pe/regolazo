import { expect, test } from "@playwright/test";

test("app shell renders", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("title")).toHaveText("ReGolazo");
});
