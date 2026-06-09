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
/**
 * @param {import("@playwright/test").Page} page
 * @param {string[]} errors
 */
async function debugSurfacePage(page, errors) {
  const status = await page.locator("[data-dft-surface-status]").innerText().catch(() => "<missing status>");
  const body = await page.locator("body").innerHTML().catch(() => "<missing body>");
  return `errors=${JSON.stringify(errors)} status=${status} body=${body.slice(0, 1800)}`;
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
    assert.match(status, /band 0/, await debugSurfacePage(page, errors));
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
    nu: 3,
    nv: 3,
    k1: [[-4, 0, 4], [-4, 0, 4], [-4, 0, 4]],
    k2: [[-4, -4, -4], [0, 0, 0], [4, 4, 4]],
    energies: [
      [[0], [1], [0]],
      [[1], [2], [1]],
      [[0], [1], [0]],
    ],
    bands: [0],
    nbands: 1,
    selected_band: 0,
    bz_hexagon: [
      [Math.PI, 0],
      [Math.PI, Math.PI],
      [0, Math.PI],
      [-Math.PI, 0],
      [-Math.PI, -Math.PI],
      [0, -Math.PI],
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
  /** @type {string[]} */
  const errors = [];

  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });

  try {
    await page.goto(server.url);
    await page.waitForSelector("dft-band-surface-viewer .band-surface-viewer-three-only", { timeout: 10000 });

    await page.waitForFunction(() => {
      const text = document.querySelector("[data-dft-surface-status]")?.textContent || "";
      return text.includes("hex mask off") && text.includes("triangles");
    }, { timeout: 10000 });

    const before = await page.locator("[data-dft-surface-status]").innerText();
    assert.match(before, /hex mask off/, await debugSurfacePage(page, errors));

    await page.locator("[data-dft-mask-to-hexagon]").evaluate((input) => {
      if (!(input instanceof HTMLInputElement)) {
        throw new Error("mask toggle is not an input");
      }
      input.checked = true;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
    });

    await page.waitForFunction(() => {
      const viewer = document.querySelector("dft-band-surface-viewer");
      return viewer?.getAttribute("data-hex-mask") === "on";
    }, { timeout: 10000 }).catch(async (error) => {
      const checked = await page.locator("[data-dft-mask-to-hexagon]").isChecked().catch(() => false);
      const afterNow = await page.locator("[data-dft-surface-status]").innerText().catch(() => "<missing>");
      throw new Error(`${error.message}; checked=${checked}; before=${before}; after=${afterNow}; ${await debugSurfacePage(page, errors)}`);
    });

    const after = await page.locator("[data-dft-surface-status]").innerText();
    assert.match(after, /hex mask on/, await debugSurfacePage(page, errors));
  } finally {
    await browser.close();
    await server.close();
  }
});



test("band surface legend toggles bands", async () => {
  const root = mkdtempSync(join(tmpdir(), "dft-local-browser-multiband-"));
  const componentPath = resolve("src/dft_local/diagnostics/static/dft-local-components.js");
  writeFileSync(join(root, "dft-local-components.js"), readFileSync(componentPath));

  const payload = {
    nu: 2,
    nv: 2,
    k1: [[0, 1], [0, 1]],
    k2: [[0, 0], [1, 1]],
    energies: [
      [[0, 10], [1, 11]],
      [[2, 12], [3, 13]],
    ],
    bands: [0, 1],
    nbands: 2,
    selected_band: 0,
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

    await page.waitForFunction(() => {
      const status = document.querySelector("[data-dft-surface-status]")?.textContent || "";
      const legend = document.querySelector("[data-dft-surface-legend]")?.textContent || "";
      return status.includes("visible 2") && legend.includes("band 0") && legend.includes("band 1");
    }, { timeout: 10000 });

    await page.locator("[data-dft-surface-legend] button", { hasText: "band 1" }).click();

    await page.waitForFunction(() => {
      const status = document.querySelector("[data-dft-surface-status]")?.textContent || "";
      const hidden = document.querySelector("[data-dft-surface-legend] .band-surface-legend-item-hidden")?.textContent || "";
      return status.includes("visible 1") && status.includes("hidden 1") && hidden.includes("band 1");
    }, { timeout: 10000 });
  } finally {
    await browser.close();
    await server.close();
  }
});



test("band surface legend does not allow hiding every band", async () => {
  const root = mkdtempSync(join(tmpdir(), "dft-local-browser-legend-last-"));
  const componentPath = resolve("src/dft_local/diagnostics/static/dft-local-components.js");
  writeFileSync(join(root, "dft-local-components.js"), readFileSync(componentPath));

  const payload = {
    nu: 2,
    nv: 2,
    k1: [[0, 1], [0, 1]],
    k2: [[0, 0], [1, 1]],
    energies: [
      [[0, 10], [1, 11]],
      [[2, 12], [3, 13]],
    ],
    bands: [0, 1],
    nbands: 2,
    selected_band: 0,
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
  const browserInstance = await chromium.launch();
  const page = await browserInstance.newPage();

  try {
    await page.goto(server.url);
    await page.waitForSelector("dft-band-surface-viewer .band-surface-viewer-three-only", { timeout: 10000 });

    await page.waitForFunction(() => {
      const status = document.querySelector("[data-dft-surface-status]")?.textContent || "";
      return status.includes("visible 2");
    }, { timeout: 10000 });

    await page.locator("[data-dft-surface-legend] button", { hasText: "band 1" }).click();

    await page.waitForFunction(() => {
      const status = document.querySelector("[data-dft-surface-status]")?.textContent || "";
      return status.includes("visible 1") && status.includes("hidden 1");
    }, { timeout: 10000 });

    const remainingDisabled = await page
      .locator("[data-dft-surface-legend] button:not(.band-surface-legend-item-hidden)")
      .isDisabled();
    assert.equal(remainingDisabled, true);
  } finally {
    await browserInstance.close();
    await server.close();
  }
});



test("generic model refresh updates existing custom element", async () => {
  const root = mkdtempSync(join(tmpdir(), "dft-local-browser-model-refresh-"));
  const componentPath = resolve("src/dft_local/diagnostics/static/dft-local-components.js");
  writeFileSync(join(root, "dft-local-components.js"), readFileSync(componentPath));

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
  <script type="application/json" id="model-test">{"value": 1}</script>
  <div id="host" data-dft-model="model-test"></div>
  <script type="module" src="/dft-local-components.js"></script>
  <script>
    window.__receivedModels = [];
    const host = document.getElementById("host");
    host.updateModel = (model) => window.__receivedModels.push(model);
  </script>
</body>
</html>`);

  const server = await serveDirectory(root);
  const browserInstance = await chromium.launch();
  const page = await browserInstance.newPage();

  try {
    await page.goto(server.url);
    await page.waitForFunction(() => typeof /** @type {any} */ (window).dftRefreshModels === "function", { timeout: 10000 });

    await page.evaluate(() => {
      /** @type {any} */ (window).dftRefreshModels?.(document);
    });

    await page.waitForFunction(() => /** @type {any} */ (window).__receivedModels?.length === 1, { timeout: 10000 });

    const first = await page.evaluate(() => /** @type {any} */ (window).__receivedModels?.[0]);
    assert.deepEqual(first, { value: 1 });

    await page.evaluate(() => {
      const script = document.getElementById("model-test");
      if (!script) throw new Error("missing model-test script");
      script.textContent = JSON.stringify({ value: 2 });
      /** @type {any} */ (window).dftRefreshModels?.(document);
    });

    await page.waitForFunction(() => /** @type {any} */ (window).__receivedModels?.length === 2, { timeout: 10000 });

    const second = await page.evaluate(() => /** @type {any} */ (window).__receivedModels?.[1]);
    assert.deepEqual(second, { value: 2 });
  } finally {
    await browserInstance.close();
    await server.close();
  }
});



test("datastar-style run patches model island without replacing viewer", async () => {
  const root = mkdtempSync(join(tmpdir(), "dft-local-browser-datastar-model-"));
  const componentPath = resolve("src/dft_local/diagnostics/static/dft-local-components.js");
  writeFileSync(join(root, "dft-local-components.js"), readFileSync(componentPath));

  const firstPayload = {
    nu: 2,
    nv: 2,
    k1: [[0, 1], [0, 1]],
    k2: [[0, 0], [1, 1]],
    energies: [
      [[0], [1]],
      [[2], [3]],
    ],
    bands: [0],
    nbands: 1,
    selected_band: 0,
  };

  const secondPayload = {
    ...firstPayload,
    energies: [
      [[10], [11]],
      [[12], [13]],
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
  <section id="dft-block-surface" data-dft-block="surface" data-dft-block-kind="json-rendered">
    <script type="application/json" id="dft-model-surface" data-dft-model="surface">${JSON.stringify(firstPayload).replaceAll("</", "<\\/")}</script>
    <dft-band-surface-viewer data-source="dft-model-surface" data-dft-model="dft-model-surface"></dft-band-surface-viewer>
  </section>
  <script type="module" src="/dft-local-components.js"></script>
</body>
</html>`);

  const server = await serveDirectory(root);
  const browserInstance = await chromium.launch();
  const page = await browserInstance.newPage();

  try {
    await page.goto(server.url);
    await page.waitForSelector("dft-band-surface-viewer .band-surface-viewer-three-only", { timeout: 10000 });

    await page.evaluate(() => {
      const viewer = document.querySelector("dft-band-surface-viewer");
      if (!viewer) throw new Error("missing viewer");
      /** @type {any} */ (window).__viewerBefore = viewer;
    });

    await page.evaluate((payload) => {
      const model = document.getElementById("dft-model-surface");
      if (!model) throw new Error("missing model");
      model.textContent = JSON.stringify(payload);
      /** @type {any} */ (window).dftRefreshModels?.(document);
    }, secondPayload);

    await page.waitForFunction(() => {
      const viewer = document.querySelector("dft-band-surface-viewer");
      return viewer && viewer === /** @type {any} */ (window).__viewerBefore;
    }, { timeout: 10000 });

    const payloadValue = await page.evaluate(() => {
      const viewer = document.querySelector("dft-band-surface-viewer");
      return /** @type {any} */ (viewer)?.payload?.energies?.[0]?.[0]?.[0];
    });

    assert.equal(payloadValue, 10);
  } finally {
    await browserInstance.close();
    await server.close();
  }
});
