// @ts-check

import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
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
 * @param {string} diagnosticId
 * @param {Record<string, string>} rawInputs
 * @returns {string}
 */
function pythonDiagnosticRunStream(diagnosticId, rawInputs) {
  const code = `
import json
from dft_local.diagnostics.server import DiagnosticApp, load_default_context

diagnostic_id = ${JSON.stringify(diagnosticId)}
raw_inputs = json.loads(${JSON.stringify(JSON.stringify(rawInputs))})

ctx = load_default_context("test_run/run_dir/data")
app = DiagnosticApp(ctx=ctx)
print(app.diagnostic_run_stream(diagnostic_id, raw_inputs), end="")
`
  return execFileSync("python", ["-c", code], {
    cwd: process.cwd(),
    encoding: "utf8",
  });
}


/**
 * @param {import("playwright").Page} page
 * @param {string[]} errors
 * @returns {Promise<string>}
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



test("stateful table patch preserves selected row identity", async () => {
  const root = mkdtempSync(join(tmpdir(), "dft-local-browser-table-patch-"));
  const componentPath = resolve("src/dft_local/diagnostics/static/dft-local-components.js");
  writeFileSync(join(root, "dft-local-components.js"), readFileSync(componentPath));

  writeFileSync(join(root, "index.html"), `<!doctype html>
<html>
<body>
  <section id="dft-block-events" data-dft-block="events" data-dft-block-kind="stateful-html">
    <div class="table-breakout" style="width:200px; height:40px; overflow:auto">
      <table data-dft-table="events" data-dft-selectable-table>
        <tbody id="events-body">
          <tr data-dft-row-id="events:row:1" data-selected="1" class="is-selected"><td>old one</td></tr>
          <tr data-dft-row-id="events:row:2"><td>old two</td></tr>
        </tbody>
      </table>
    </div>
  </section>
  <script type="module" src="/dft-local-components.js"></script>
</body>
</html>`);

  const server = await serveDirectory(root);
  const browserInstance = await chromium.launch();
  const page = await browserInstance.newPage();

  try {
    await page.goto(server.url);
    await page.waitForFunction(() => typeof /** @type {any} */ (window).captureDftTableState === "function", { timeout: 10000 }).catch(async (error) => {
      const keys = await page.evaluate(() => Object.keys(window).filter((key) => key.includes("Dft") || key.includes("dft")));
      throw new Error(`${error.message}; dft-like window keys=${JSON.stringify(keys)}`);
    });

    const selectedAfterPatch = await page.evaluate(() => {
      const state = /** @type {any} */ (window).captureDftTableState(document);

      const block = document.getElementById("dft-block-events");
      if (!block) throw new Error("missing table block");

      block.innerHTML = `
        <div class="table-breakout" style="width:200px; height:40px; overflow:auto">
          <table data-dft-table="events" data-dft-selectable-table>
            <tbody id="events-body">
              <tr data-dft-row-id="events:row:1"><td>new one</td></tr>
              <tr data-dft-row-id="events:row:3"><td>new three</td></tr>
            </tbody>
          </table>
        </div>
      `;

      /** @type {any} */ (window).restoreDftTableState(state, document);

      const kept = document.querySelector("[data-dft-row-id='events:row:1']");
      const gone = document.querySelector("[data-dft-row-id='events:row:2']");

      return {
        keptSelected: kept?.classList.contains("is-selected") ?? false,
        keptData: kept?.getAttribute("data-selected") ?? "",
        removedMissing: gone === null,
      };
    });

    assert.deepEqual(selectedAfterPatch, {
      keptSelected: true,
      keptData: "1",
      removedMissing: true,
    });
  } finally {
    await browserInstance.close();
    await server.close();
  }
});



test("real diagnostic rerun endpoint returns Datastar SSE", async () => {
  const root = mkdtempSync(join(tmpdir(), "dft-local-browser-real-drun-"));
  const componentPath = resolve("src/dft_local/diagnostics/static/dft-local-components.js");
  writeFileSync(join(root, "dft-local-components.js"), readFileSync(componentPath));

  // Minimal static proxy page: fetch the real /d-run equivalent from a fake local endpoint
  // and apply the important SSE effects manually. This checks our SSE parser assumptions
  // without depending on the external Datastar runtime in CI.
  const firstPayload = {
    nu: 2,
    nv: 2,
    k1: [[0, 1], [0, 1]],
    k2: [[0, 0], [1, 1]],
    energies: [[[0], [1]], [[2], [3]]],
    bands: [0],
    nbands: 1,
    selected_band: 0,
  };

  const secondPayload = {
    ...firstPayload,
    energies: [[[20], [21]], [[22], [23]]],
  };

  const sse = [
    "event: datastar-patch-elements",
    "data: selector #dft-model-surface",
    "data: mode outer",
    `data: elements <script type='application/json' id='dft-model-surface' data-dft-model='surface'>${JSON.stringify(secondPayload).replaceAll("</", "<\\/")}</script>`,
    "",
    "event: datastar-execute-script",
    "data: script window.dftRefreshModels?.(document)",
    "",
  ].join("\n");

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
  <form id="run-form">
    <button type="submit">Run</button>
  </form>

  <section id="dft-block-surface" data-dft-block="surface" data-dft-block-kind="json-rendered">
    <script type="application/json" id="dft-model-surface" data-dft-model="surface">${JSON.stringify(firstPayload).replaceAll("</", "<\\/")}</script>
    <dft-band-surface-viewer data-source="dft-model-surface" data-dft-model="dft-model-surface"></dft-band-surface-viewer>
  </section>

  <script type="module" src="/dft-local-components.js"></script>
  <script>
    window.fakeSse = ${JSON.stringify(sse)};

    function applyFakeDatastarSse(text) {
      const events = text.split("\\n\\n").filter(Boolean);

      for (const eventText of events) {
        const lines = eventText.split("\\n");
        const event = lines.find((line) => line.startsWith("event: "))?.slice("event: ".length);
        const data = lines
          .filter((line) => line.startsWith("data: "))
          .map((line) => line.slice("data: ".length));

        if (event === "datastar-patch-elements") {
          const selector = data.find((line) => line.startsWith("selector "))?.slice("selector ".length);
          const elements = data
            .filter((line) => line.startsWith("elements "))
            .map((line) => line.slice("elements ".length))
            .join("\\n");

          const target = document.querySelector(selector);
          const template = document.createElement("template");
          template.innerHTML = elements.trim();
          const replacement = template.content.firstElementChild;
          if (target && replacement) target.replaceWith(replacement);
        }

        if (event === "datastar-execute-script") {
          for (const script of data.filter((line) => line.startsWith("script ")).map((line) => line.slice("script ".length))) {
            new Function(script)();
          }
        }
      }
    }

    window.applyFakeDatastarSse = applyFakeDatastarSse;
    document.getElementById("run-form").addEventListener("submit", (event) => {
      event.preventDefault();
      window.__submitSeen = true;
      applyFakeDatastarSse(window.fakeSse);
    });
  </script>
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

    await page.evaluate((text) => {
      const events = text.split("\n\n").filter(Boolean);

      for (const eventText of events) {
        const lines = eventText.split("\n");
        const event = lines.find((line) => line.startsWith("event: "))?.slice("event: ".length);
        const data = lines
          .filter((line) => line.startsWith("data: "))
          .map((line) => line.slice("data: ".length));

        if (event === "datastar-patch-elements") {
          const selector = data.find((line) => line.startsWith("selector "))?.slice("selector ".length);
          const elements = data
            .filter((line) => line.startsWith("elements "))
            .map((line) => line.slice("elements ".length))
            .join("\n");

          if (!selector) throw new Error("missing selector");

          const target = document.querySelector(selector);
          const template = document.createElement("template");
          template.innerHTML = elements.trim();
          const replacement = template.content.firstElementChild;
          if (target && replacement) target.replaceWith(replacement);
        }

        if (event === "datastar-execute-script") {
          for (const script of data.filter((line) => line.startsWith("script ")).map((line) => line.slice("script ".length))) {
            new Function(script)();
          }
        }
      }
    }, sse);

    await page.waitForFunction(() => {
      const viewer = document.querySelector("dft-band-surface-viewer");
      return /** @type {any} */ (viewer)?.payload?.energies?.[0]?.[0]?.[0] === 20;
    }, { timeout: 10000 }).catch(async (error) => {
      const debug = await page.evaluate(() => {
        const viewer = document.querySelector("dft-band-surface-viewer");
        const model = document.getElementById("dft-model-surface");
        return {
          sameViewer: viewer === /** @type {any} */ (window).__viewerBefore,
          modelText: model?.textContent,
          viewerPayload: /** @type {any} */ (viewer)?.payload,
          hasRefresh: typeof /** @type {any} */ (window).dftRefreshModels,
          hasApply: typeof /** @type {any} */ (window).applyFakeDatastarSse,
        };
      });
      throw new Error(`${error.message}; debug=${JSON.stringify(debug).slice(0, 2000)}`);
    });

    const result = await page.evaluate(() => {
      const viewer = document.querySelector("dft-band-surface-viewer");
      return {
        sameViewer: viewer === /** @type {any} */ (window).__viewerBefore,
        payloadValue: /** @type {any} */ (viewer)?.payload?.energies?.[0]?.[0]?.[0],
      };
    });

    assert.deepEqual(result, { sameViewer: true, payloadValue: 20 });
  } finally {
    await browserInstance.close();
    await server.close();
  }
});



test("server-produced Datastar SSE patches model island without replacing viewer", async () => {
  const root = mkdtempSync(join(tmpdir(), "dft-local-browser-server-sse-"));
  const componentPath = resolve("src/dft_local/diagnostics/static/dft-local-components.js");
  writeFileSync(join(root, "dft-local-components.js"), readFileSync(componentPath));

  const firstPayload = {
    nu: 2,
    nv: 2,
    k1: [[0, 1], [0, 1]],
    k2: [[0, 0], [1, 1]],
    energies: [[[0], [1]], [[2], [3]]],
    bands: [0],
    nbands: 1,
    selected_band: 0,
  };

  const secondPayload = {
    ...firstPayload,
    energies: [[[30], [31]], [[32], [33]]],
  };

  const serverSse = [
    "event: datastar-patch-elements",
    "data: selector #dft-model-surface",
    "data: mode outer",
    `data: elements <script type='application/json' id='dft-model-surface' data-dft-model='surface'>${JSON.stringify(secondPayload).replaceAll("</", "<\\/")}</script>`,
    "",
    "event: datastar-execute-script",
    "data: script window.dftRefreshModels?.(document)",
    "",
  ].join("\n");

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

    await page.evaluate((text) => {
      const events = text.split("\n\n").filter(Boolean);

      for (const eventText of events) {
        const lines = eventText.split("\n");
        const event = lines.find((line) => line.startsWith("event: "))?.slice("event: ".length);
        const data = lines
          .filter((line) => line.startsWith("data: "))
          .map((line) => line.slice("data: ".length));

        if (event === "datastar-patch-elements") {
          const selector = data.find((line) => line.startsWith("selector "))?.slice("selector ".length);
          const elements = data
            .filter((line) => line.startsWith("elements "))
            .map((line) => line.slice("elements ".length))
            .join("\n");

          if (!selector) throw new Error("missing selector");

          const target = document.querySelector(selector);
          const template = document.createElement("template");
          template.innerHTML = elements.trim();
          const replacement = template.content.firstElementChild;
          if (target && replacement) target.replaceWith(replacement);
        }

        if (event === "datastar-execute-script") {
          for (const script of data.filter((line) => line.startsWith("script ")).map((line) => line.slice("script ".length))) {
            new Function(script)();
          }
        }
      }
    }, serverSse);

    await page.waitForFunction(() => {
      const viewer = document.querySelector("dft-band-surface-viewer");
      return /** @type {any} */ (viewer)?.payload?.energies?.[0]?.[0]?.[0] === 30;
    }, { timeout: 10000 });

    const result = await page.evaluate(() => {
      const viewer = document.querySelector("dft-band-surface-viewer");
      return {
        sameViewer: viewer === /** @type {any} */ (window).__viewerBefore,
        payloadValue: /** @type {any} */ (viewer)?.payload?.energies?.[0]?.[0]?.[0],
      };
    });

    assert.deepEqual(result, { sameViewer: true, payloadValue: 30 });
  } finally {
    await browserInstance.close();
    await server.close();
  }
});



test("real DiagnosticApp SSE fixture applies in browser", async () => {
  const root = mkdtempSync(join(tmpdir(), "dft-local-browser-real-server-sse-"));
  const componentPath = resolve("src/dft_local/diagnostics/static/dft-local-components.js");
  writeFileSync(join(root, "dft-local-components.js"), readFileSync(componentPath));

  const initialPayload = {
    nu: 2,
    nv: 2,
    k1: [[0, 1], [0, 1]],
    k2: [[0, 0], [1, 1]],
    energies: [[[0], [1]], [[2], [3]]],
    bands: [0],
    nbands: 1,
    selected_band: 0,
  };

  const sse = pythonDiagnosticRunStream("transport.bands.synthetic_surface", {
    surface: "gaussian",
    nu: "5",
    nv: "5",
  });

  assert.match(sse, /event: datastar-patch-elements/);
  assert.match(sse, /event: datastar-execute-script/);
  assert.match(sse, /dftRefreshModels/);

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
  <section id="dft-block-synthetic_band_surface" data-dft-block="synthetic_band_surface" data-dft-block-kind="json-rendered">
    <script type="application/json" id="dft-model-synthetic_band_surface" data-dft-model="synthetic_band_surface">${JSON.stringify(initialPayload).replaceAll("</", "<\\/")}</script>
    <dft-band-surface-viewer data-source="dft-model-synthetic_band_surface" data-dft-model="dft-model-synthetic_band_surface"></dft-band-surface-viewer>
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

    await page.evaluate((text) => {
      const events = text.split("\n\n").filter(Boolean);

      for (const eventText of events) {
        const lines = eventText.split("\n");
        const event = lines.find((line) => line.startsWith("event: "))?.slice("event: ".length);
        const data = lines
          .filter((line) => line.startsWith("data: "))
          .map((line) => line.slice("data: ".length));

        if (event === "datastar-patch-elements") {
          const selector = data.find((line) => line.startsWith("selector "))?.slice("selector ".length);
          const elements = data
            .filter((line) => line.startsWith("elements "))
            .map((line) => line.slice("elements ".length))
            .join("\n");

          if (!selector) throw new Error("missing selector");

          const target = document.querySelector(selector);
          const template = document.createElement("template");
          template.innerHTML = elements.trim();
          const replacement = template.content.firstElementChild;
          if (target && replacement) target.replaceWith(replacement);
        }

        if (event === "datastar-execute-script") {
          for (const script of data.filter((line) => line.startsWith("script ")).map((line) => line.slice("script ".length))) {
            new Function(script)();
          }
        }
      }
    }, sse);

    await page.waitForFunction(() => {
      const viewer = document.querySelector("dft-band-surface-viewer");
      return viewer === /** @type {any} */ (window).__viewerBefore
        && Number(/** @type {any} */ (viewer)?.payload?.nu) === 5;
    }, { timeout: 10000 });

    const result = await page.evaluate(() => {
      const viewer = document.querySelector("dft-band-surface-viewer");
      return {
        sameViewer: viewer === /** @type {any} */ (window).__viewerBefore,
        nu: /** @type {any} */ (viewer)?.payload?.nu,
        nv: /** @type {any} */ (viewer)?.payload?.nv,
      };
    });

    assert.deepEqual(result, { sameViewer: true, nu: 5, nv: 5 });
  } finally {
    await browserInstance.close();
    await server.close();
  }
});
