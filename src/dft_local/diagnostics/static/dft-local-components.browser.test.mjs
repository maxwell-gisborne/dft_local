// @ts-check

import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createServer } from "node:http";

let chromium;
try {
  ({ chromium } = await import("@playwright/test"));
} catch {
  throw new Error("Missing browser test dependency. Run: npm install");
}

/**
 * @param {string} root
 * @returns {Promise<{url:string, close:() => Promise<void>}>}
 */
async function serveDirectory(root) {
  const server = createServer((request, response) => {
    const url = new URL(request.url || "/", "http://127.0.0.1");
    const pathname = url.pathname === "/" ? "/index.html" : url.pathname;
    const file = join(root, pathname);

    try {
      const body = readFileSync(file);
      response.writeHead(200, {
        "content-type": file.endsWith(".js") ? "text/javascript" : "text/html",
      });
      response.end(body);
    } catch {
      response.writeHead(404);
      response.end("not found");
    }
  });

  await new Promise((resolveListen) => {
    server.listen(0, "127.0.0.1", () => resolveListen(undefined));
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("server failed");

  return {
    url: `http://127.0.0.1:${address.port}/`,
    close: () => new Promise((resolveClose) => server.close(() => resolveClose())),
  };
}

test("band surface viewer renders in a real browser", async () => {
  const root = mkdtempSync(join(tmpdir(), "dft-local-browser-"));
  const componentPath = resolve("src/dft_local/diagnostics/static/dft-local-components.js");
  writeFileSync(join(root, "dft-local-components.js"), readFileSync(componentPath));

  const payload = {
    nu: 3,
    nv: 3,
    k1: [
      [-1, 0, 1],
      [-1, 0, 1],
      [-1, 0, 1],
    ],
    k2: [
      [-1, -1, -1],
      [0, 0, 0],
      [1, 1, 1],
    ],
    mask: [
      [true, true, true],
      [true, true, true],
      [true, true, true],
    ],
    energies: [
      [[0], [1], [0]],
      [[1], [2], [1]],
      [[0], [1], [0]],
    ],
    bands: [0],
    nbands: 1,
    selected_band: 0,
    bz_hexagon: [
      [1, 0],
      [0.5, 0.866],
      [-0.5, 0.866],
      [-1, 0],
      [-0.5, -0.866],
      [0.5, -0.866],
    ],
  };

  writeFileSync(join(root, "index.html"), `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three@0.160.0/build/three.module.js"
    }
  }
  </script>
</head>
<body>
  <script type="application/json" id="surface-payload">${JSON.stringify(payload).replaceAll("</", "<\\/")}</script>
  <dft-band-surface-viewer data-source="surface-payload"></dft-band-surface-viewer>
  <script type="module" src="/dft-local-components.js"></script>
</body>
</html>`);

  const server = await serveDirectory(root);
  const browser = await chromium.launch();
  const page = await browser.newPage();

  /** @type {string[]} */
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });

  try {
    await page.goto(server.url);
    await page.waitForSelector("dft-band-surface-viewer .band-surface-viewer-three-only", { timeout: 10000 });
    await page.waitForSelector("[data-dft-three-surface]", { timeout: 10000 });

    const status = await page.locator("[data-dft-surface-status]").innerText();
    assert.match(status, /band 0/);
    assert.match(status, /vertices 9/);
    assert.match(status, /hex mask off/);
    assert.doesNotMatch(status, /no surface data/);

    await page.waitForFunction(() => {
      const host = document.querySelector("[data-dft-three-surface]");
      if (!host) return false;

      return (
        host.querySelector("canvas") !== null
        || (host.textContent || "").includes("three.js failed to load")
        || (host.textContent || "").includes("no surface data")
      );
    }, { timeout: 10000 });

    const threeText = await page.locator("[data-dft-three-surface]").innerText();
    const canvasCount = await page.locator("[data-dft-three-surface] canvas").count();

    assert.equal(
      canvasCount > 0 || threeText.includes("three.js failed to load"),
      true,
      `expected a three.js canvas or explicit load failure, got text=${threeText}`,
    );

    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    await server.close();
  }
});



test("band surface hex mask toggle changes rendered status", async () => {
  const root = mkdtempSync(join(tmpdir(), "dft-local-browser-mask-"));
  const componentPath = resolve("src/dft_local/diagnostics/static/dft-local-components.js");
  writeFileSync(join(root, "dft-local-components.js"), readFileSync(componentPath));

  const payload = {
    nu: 2,
    nv: 2,
    k1: [[0, 10], [0, 10]],
    k2: [[0, 0], [1, 1]],
    energies: [
      [[0], [1]],
      [[2], [3]],
    ],
    bands: [0],
    nbands: 1,
    selected_band: 0,
    bz_hexagon: [
      [2, 0],
      [1, 1],
      [-1, 1],
      [-2, 0],
      [-1, -1],
      [1, -1],
    ],
  };

  writeFileSync(join(root, "index.html"), `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three@0.160.0/build/three.module.js"
    }
  }
  </script>
</head>
<body>
  <script type="application/json" id="surface-payload">${JSON.stringify(payload).replaceAll("</", "<\/")}</script>
  <dft-band-surface-viewer data-source="surface-payload"></dft-band-surface-viewer>
  <script type="module" src="/dft-local-components.js"></script>
</body>
</html>`);

  const server = await serveDirectory(root);
  const browser = await chromium.launch();
  const page = await browser.newPage();

  try {
    await page.goto(server.url);
    await page.waitForSelector("dft-band-surface-viewer .band-surface-viewer-three-only", { timeout: 10000 });

    const status = page.locator("[data-dft-surface-status]");
    await page.waitForFunction(() => {
      const el = document.querySelector("[data-dft-surface-status]");
      return (el?.textContent || "").includes("triangles 2")
        && (el?.textContent || "").includes("hex mask off");
    }, { timeout: 10000 });

    await page.locator("[data-dft-mask-to-hexagon]").check();

    await page.waitForFunction(() => {
      const el = document.querySelector("[data-dft-surface-status]");
      return (el?.textContent || "").includes("triangles 0")
        && (el?.textContent || "").includes("hex mask on");
    }, { timeout: 10000 });

    assert.match(await status.innerText(), /hex mask on/);
    assert.match(await status.innerText(), /triangles 0/);
  } finally {
    await browser.close();
    await server.close();
  }
});
