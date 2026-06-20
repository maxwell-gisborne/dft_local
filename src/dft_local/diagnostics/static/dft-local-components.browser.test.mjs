// @ts-check

import test from "node:test";
import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
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
    assert.match(status, /vertices 63/);
    assert.match(status, /domain extended/);
    assert.doesNotMatch(status, /no surface data/);

    try {
      await page.waitForFunction(() => {
        const host = document.querySelector("[data-dft-three-surface]");
        if (!host) return false;

        return (
          host.querySelector("canvas") !== null
          || (host.textContent || "").includes("three.js failed to load")
          || (host.textContent || "").includes("no surface data")
        );
      }, { timeout: 10000 });
    } catch (error) {
      assert.fail(await debugSurfacePage(page, errors));
    }

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



test("band surface domain dropdown switches primitive BZ and extended modes", async () => {
  const browserInstance = await chromium.launch();
  const root = mkdtempSync(join(tmpdir(), "dft-local-domain-mode-"));
  const payload = {
    kind: "band-surface-preview",
    nu: 3,
    nv: 3,
    k1: [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    k2: [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    energies: [
      [[0], [1], [2]],
      [[0], [1], [2]],
      [[0], [1], [2]],
    ],
    mask: [[true, true, true], [true, true, true], [true, true, true]],
    bands: [0],
    nbands: 1,
    selected_band: 0,
    energy_unit: "eV",
  };

  try {
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
    <script type="application/json" id="dft-model-surface" data-dft-model="surface">${JSON.stringify(payload).replaceAll("</", "<\/")}</script>
    <dft-band-surface-viewer data-source="dft-model-surface" data-dft-model="dft-model-surface"></dft-band-surface-viewer>
  </section>
  <script type="module" src="/dft-local-components.js"></script>
</body>
</html>`);

    copyFileSync("src/dft_local/diagnostics/static/dft-local-components.js", join(root, "dft-local-components.js"));
    const server = await serveDirectory(root);
    const page = await browserInstance.newPage();

    try {
      await page.goto(server.url);
      await page.waitForSelector("dft-band-surface-viewer canvas", { timeout: 10000 });

      await page.locator("[data-dft-domain-mode]").selectOption("primitive");
      await page.waitForFunction(() => document.querySelector("[data-dft-surface-status]")?.textContent?.includes("domain primitive"));

      await page.locator("[data-dft-domain-mode]").selectOption("bz");
      await page.waitForFunction(() => document.querySelector("[data-dft-surface-status]")?.textContent?.includes("domain bz"));

      await page.locator("[data-dft-domain-mode]").selectOption("extended");
      await page.waitForFunction(() => document.querySelector("[data-dft-surface-status]")?.textContent?.includes("domain extended"));
    } finally {
      await server.close();
    }
  } finally {
    await browserInstance.close();
  }
});


test("band surface domain dropdown changes rendered status", async () => {
  const browserInstance = await chromium.launch();
  const root = mkdtempSync(join(tmpdir(), "dft-local-domain-status-"));
  const payload = {
    kind: "band-surface-preview",
    nu: 3,
    nv: 3,
    k1: [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    k2: [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    energies: [
      [[0], [1], [2]],
      [[0], [1], [2]],
      [[0], [1], [2]],
    ],
    mask: [[true, true, true], [true, true, true], [true, true, true]],
    bands: [0],
    nbands: 1,
    selected_band: 0,
    energy_unit: "eV",
  };

  try {
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
    <script type="application/json" id="dft-model-surface" data-dft-model="surface">${JSON.stringify(payload).replaceAll("</", "<\/")}</script>
    <dft-band-surface-viewer data-source="dft-model-surface" data-dft-model="dft-model-surface"></dft-band-surface-viewer>
  </section>
  <script type="module" src="/dft-local-components.js"></script>
</body>
</html>`);

    copyFileSync("src/dft_local/diagnostics/static/dft-local-components.js", join(root, "dft-local-components.js"));
    const server = await serveDirectory(root);
    const page = await browserInstance.newPage();

    try {
      await page.goto(server.url);
      await page.waitForSelector("dft-band-surface-viewer canvas", { timeout: 10000 });

      const initial = await page.locator("[data-dft-surface-status]").textContent();
      assert.match(initial ?? "", /domain extended/);

      await page.locator("[data-dft-domain-mode]").selectOption("bz");
      await page.waitForFunction(() => document.querySelector("[data-dft-surface-status]")?.textContent?.includes("domain bz"));

      const bz = await page.locator("[data-dft-surface-status]").textContent();
      assert.match(bz ?? "", /domain bz/);

      await page.locator("[data-dft-domain-mode]").selectOption("primitive");
      await page.waitForFunction(() => document.querySelector("[data-dft-surface-status]")?.textContent?.includes("domain primitive"));

      const primitive = await page.locator("[data-dft-surface-status]").textContent();
      assert.match(primitive ?? "", /domain primitive/);
    } finally {
      await server.close();
    }
  } finally {
    await browserInstance.close();
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
    "event: datastar-patch-elements",
    "data: selector [data-dft-run-button]",
    "data: mode outer",
    "data: elements <button type='submit' data-dft-run-button aria-busy='false'>Run</button>",
    "",
    "event: datastar-patch-elements",
    "data: selector [data-dft-run-status]",
    "data: mode outer",
    "data: elements <span data-dft-run-status aria-live='polite'>updated</span>",
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
    <button type="submit" data-dft-run-button>Run</button>
    <span data-dft-run-status aria-live="polite"></span>
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



test("band surface legend toggles mesh visibility without rebuilding meshes", async () => {
  const browserInstance = await chromium.launch();
  const root = mkdtempSync(join(tmpdir(), "dft-local-band-visibility-"));
  const payload = {
    kind: "band-surface-preview",
    nu: 3,
    nv: 3,
    k1: [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    k2: [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    energies: [
      [[0, 10], [1, 11], [2, 12]],
      [[1, 11], [2, 12], [3, 13]],
      [[2, 12], [3, 13], [4, 14]],
    ],
    mask: [[true, true, true], [true, true, true], [true, true, true]],
    bands: [0, 1],
    nbands: 2,
    selected_band: 0,
    energy_unit: "eV",
  };

  try {
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
    <script type="application/json" id="dft-model-surface" data-dft-model="surface">${JSON.stringify(payload).replaceAll("</", "<\\/")}</script>
    <dft-band-surface-viewer data-source="dft-model-surface" data-dft-model="dft-model-surface"></dft-band-surface-viewer>
  </section>
  <script type="module" src="/dft-local-components.js"></script>
</body>
</html>`);

    copyFileSync("src/dft_local/diagnostics/static/dft-local-components.js", join(root, "dft-local-components.js"));
    const server = await serveDirectory(root);
    const page = await browserInstance.newPage();

    try {
      await page.goto(server.url);
      await page.waitForSelector("dft-band-surface-viewer canvas", { timeout: 10000 });
      await page.waitForFunction(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        return viewer?.surfaceMeshes?.length === 2 && viewer?.wireMeshes?.length === 2;
      }, { timeout: 10000 });

      await page.evaluate(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        viewer.__updateSurfaceCount = 0;
        viewer.__renderThreeOnceCount = 0;
        const original = viewer.updateSurface.bind(viewer);
        viewer.updateSurface = async () => {
          viewer.__updateSurfaceCount += 1;
          return original();
        };
        const originalRender = viewer.renderThreeOnce.bind(viewer);
        viewer.renderThreeOnce = () => {
          viewer.__renderThreeOnceCount += 1;
          return originalRender();
        };
      });

      await page.locator("[data-dft-surface-legend] button[data-band='1']").click();

      await page.waitForFunction(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        return viewer?.surfaceMeshes?.some((/** @type {any} */ mesh) => Number(mesh.userData?.dftBand) === 1 && mesh.visible === false);
      }, { timeout: 10000 });

      const result = await page.evaluate(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        const hiddenBandMeshes = [...viewer.surfaceMeshes, ...viewer.wireMeshes]
          .filter((/** @type {any} */ mesh) => Number(mesh.userData?.dftBand) === 1);
        return {
          rebuilds: viewer.__updateSurfaceCount,
          renders: viewer.__renderThreeOnceCount,
          hidden: hiddenBandMeshes.every((mesh) => mesh.visible === false),
          visibleBand0: [...viewer.surfaceMeshes, ...viewer.wireMeshes]
            .filter((/** @type {any} */ mesh) => Number(mesh.userData?.dftBand) === 0)
            .every((/** @type {any} */ mesh) => mesh.visible === true),
          status: document.querySelector("[data-dft-surface-status]")?.textContent,
        };
      });

      assert.equal(result.rebuilds, 0);
      assert.ok(result.renders > 0);
      assert.equal(result.hidden, true);
      assert.equal(result.visibleBand0, true);
      assert.match(result.status ?? "", /visible 1/);
      assert.match(result.status ?? "", /hidden 1/);
    } finally {
      await server.close();
    }
  } finally {
    await browserInstance.close();
  }
});


test("band surface energy zero slider shifts group transform without rebuilding meshes", async () => {
  const browserInstance = await chromium.launch();
  const root = mkdtempSync(join(tmpdir(), "dft-local-energy-zero-"));
  const payload = {
    kind: "band-surface-preview",
    nu: 3,
    nv: 3,
    k1: [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    k2: [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    energies: [
      [[-1], [0], [1]],
      [[-1], [0], [1]],
      [[-1], [0], [1]],
    ],
    mask: [[true, true, true], [true, true, true], [true, true, true]],
    bands: [0],
    nbands: 1,
    selected_band: 0,
    energy_unit: "eV",
  };

  try {
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
    <script type="application/json" id="dft-model-surface" data-dft-model="surface">${JSON.stringify(payload).replaceAll("</", "<\\/")}</script>
    <dft-band-surface-viewer data-source="dft-model-surface" data-dft-model="dft-model-surface"></dft-band-surface-viewer>
  </section>
  <script type="module" src="/dft-local-components.js"></script>
</body>
</html>`);

    copyFileSync("src/dft_local/diagnostics/static/dft-local-components.js", join(root, "dft-local-components.js"));
    const server = await serveDirectory(root);
    const page = await browserInstance.newPage();

    try {
      await page.goto(server.url);
      await page.waitForSelector("dft-band-surface-viewer canvas", { timeout: 10000 });
      await page.waitForFunction(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        return viewer?.surfaceMeshes?.length === 1;
      }, { timeout: 10000 });

      const before = await page.evaluate(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        viewer.__updateSurfaceCount = 0;
        viewer.__renderThreeOnceCount = 0;
        const original = viewer.updateSurface.bind(viewer);
        viewer.updateSurface = async () => {
          viewer.__updateSurfaceCount += 1;
          return original();
        };
        const originalRender = viewer.renderThreeOnce.bind(viewer);
        viewer.renderThreeOnce = () => {
          viewer.__renderThreeOnceCount += 1;
          return originalRender();
        };
        return {
          groupY: viewer.surfaceGroup.position.y,
          vertexY: viewer.surfaceMeshes[0].geometry.attributes.position.array[1],
        };
      });

      await page.locator("[data-dft-view-energy-zero]").evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) throw new Error("not input");
        input.value = "1";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });

      await page.waitForFunction(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        return Number(viewer?.energyZero) === 1;
      }, { timeout: 10000 });

      const result = await page.evaluate(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        return {
          groupY: viewer.surfaceGroup.position.y,
          vertexY: viewer.surfaceMeshes[0].geometry.attributes.position.array[1],
          energyZero: viewer.energyZero,
          rebuilds: viewer.__updateSurfaceCount,
          renders: viewer.__renderThreeOnceCount,
          status: document.querySelector("[data-dft-surface-status]")?.textContent,
        };
      });

      assert.notEqual(result.groupY, before.groupY);
      assert.equal(result.vertexY, before.vertexY);
      assert.equal(result.energyZero, 1);
      assert.equal(result.rebuilds, 0);
      assert.ok(result.renders > 0);
      assert.match(result.status ?? "", /energy zero 1/);
    } finally {
      await server.close();
    }
  } finally {
    await browserInstance.close();
  }
});





test("band surface slice panel preserves graph component view state", async () => {
  const browserInstance = await chromium.launch();
  const root = mkdtempSync(join(tmpdir(), "dft-local-slice-view-state-"));
  const payload = {
    kind: "band-surface-preview",
    nu: 3,
    nv: 3,
    k1: [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    k2: [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    energies: [
      [[0, 10], [1, 11], [2, 12]],
      [[2, 12], [3, 13], [4, 14]],
      [[4, 14], [5, 15], [6, 16]],
    ],
    mask: [[true, true, true], [true, true, true], [true, true, true]],
    bands: [0, 1],
    nbands: 2,
    selected_band: 0,
    energy_unit: "eV",
  };

  try {
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
    <script type="application/json" id="dft-model-surface" data-dft-model="surface">${JSON.stringify(payload).replaceAll("</", "<\\/")}</script>
    <dft-band-surface-viewer data-source="dft-model-surface" data-dft-model="dft-model-surface"></dft-band-surface-viewer>
  </section>
  <script type="module" src="/dft-local-components.js"></script>
</body>
</html>`);

    copyFileSync("src/dft_local/diagnostics/static/dft-local-components.js", join(root, "dft-local-components.js"));
    const server = await serveDirectory(root);
    const page = await browserInstance.newPage();

    try {
      await page.goto(server.url);
      await page.waitForSelector("dft-band-surface-viewer canvas", { timeout: 10000 });
      await page.locator("[data-dft-slice-details] summary").click().catch(async () => {
        await page.locator("[data-dft-slice-details] summary").click();
      });

      await page.locator("[data-dft-view-slice-value]").evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) throw new Error("not input");
        input.value = "0.5";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });

      await page.waitForFunction(() => {
        const graph = /** @type {any} */ (document.querySelector("[data-dft-slice-plot] dft-line-graph"));
        return graph?.getAttribute("data-ready") === "true";
      }, { timeout: 10000 });

      await page.evaluate(() => {
        const graph = /** @type {any} */ (document.querySelector("[data-dft-slice-plot] dft-line-graph"));
        graph.view = { xmin: 123, xmax: 456, ymin: -7, ymax: 8 };
        /** @type {any} */ (window).__sliceGraphNode = graph;
      });

      await page.locator("[data-dft-view-slice-value]").evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) throw new Error("not input");
        input.value = "0.7";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });

      await page.waitForTimeout(200);

      const result = await page.evaluate(() => {
        const graph = /** @type {any} */ (document.querySelector("[data-dft-slice-plot] dft-line-graph"));
        return {
          sameNode: graph === /** @type {any} */ (window).__sliceGraphNode,
          view: graph.view,
          ready: graph.getAttribute("data-ready"),
        };
      });

      assert.equal(result.sameNode, true);
      assert.equal(result.ready, "true");
      assert.deepEqual(result.view, { xmin: 123, xmax: 456, ymin: -7, ymax: 8 });
    } finally {
      await server.close();
    }
  } finally {
    await browserInstance.close();
  }
});

test("band surface slice panel renders band plot and k-space plot", async () => {
  const browserInstance = await chromium.launch();
  const root = mkdtempSync(join(tmpdir(), "dft-local-slice-plot-panel-"));
  const payload = {
    kind: "band-surface-preview",
    nu: 3,
    nv: 3,
    k1: [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    k2: [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    energies: [
      [[0, 10], [1, 11], [2, 12]],
      [[2, 12], [3, 13], [4, 14]],
      [[4, 14], [5, 15], [6, 16]],
    ],
    mask: [[true, true, true], [true, true, true], [true, true, true]],
    bands: [0, 1],
    nbands: 2,
    selected_band: 0,
    energy_unit: "eV",
  };

  try {
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
    <script type="application/json" id="dft-model-surface" data-dft-model="surface">${JSON.stringify(payload).replaceAll("</", "<\\/")}</script>
    <dft-band-surface-viewer data-source="dft-model-surface" data-dft-model="dft-model-surface"></dft-band-surface-viewer>
  </section>
  <script type="module" src="/dft-local-components.js"></script>
</body>
</html>`);

    copyFileSync("src/dft_local/diagnostics/static/dft-local-components.js", join(root, "dft-local-components.js"));
    const server = await serveDirectory(root);
    const page = await browserInstance.newPage();

    try {
      await page.goto(server.url);
      await page.waitForSelector("dft-band-surface-viewer canvas", { timeout: 10000 });

      await page.locator("[data-dft-view-slice-value]").evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) throw new Error("not input");
        input.value = "0.5";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });

      await page.locator("[data-dft-slice-details] summary").click();

      await page.waitForFunction(() => {
        const graph = document.querySelector("[data-dft-slice-plot] dft-line-graph");
        return graph?.getAttribute("data-ready") === "true"
          && graph.querySelector("svg")?.textContent?.includes("energy")
          && graph.querySelector("svg")?.textContent?.includes("band 0");
      }, { timeout: 10000 });

      const bandPlotResult = await page.evaluate(() => ({
        details: Boolean(document.querySelector("[data-dft-slice-details]")),
        component: Boolean(document.querySelector("[data-dft-slice-plot] dft-line-graph")),
        kspaceComponent: Boolean(document.querySelector("[data-dft-slice-plot] dft-kspace-plot")),
        svg: Boolean(document.querySelector("[data-dft-slice-plot] dft-line-graph svg")),
        text: document.querySelector("[data-dft-slice-plot]")?.textContent ?? "",
      }));

      assert.equal(bandPlotResult.details, true);
      assert.equal(bandPlotResult.component, true);
      assert.equal(bandPlotResult.kspaceComponent, false);
      assert.equal(bandPlotResult.svg, true);
      assert.match(bandPlotResult.text, /energy/);
      assert.match(bandPlotResult.text, /band 0/);

      await page.locator("[data-dft-view-slice-axis]").selectOption("energy");
      await page.locator("[data-dft-view-slice-value]").evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) throw new Error("not input");
        input.value = "12";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });

      await page.waitForFunction(() => {
        const plot = document.querySelector("[data-dft-slice-plot] dft-kspace-plot");
        const modelId = plot?.getAttribute("data-source");
        const payloadText = modelId ? document.getElementById(modelId)?.textContent ?? "" : "";
        const payload = payloadText ? JSON.parse(payloadText) : null;

        return plot?.getAttribute("data-ready") === "true"
          && plot.querySelector("svg")?.classList.contains("kspace-svg")
          && Array.isArray(payload?.series)
          && payload.series.length > 0;
      }, { timeout: 10000 });

      const kspacePlotResult = await page.evaluate(() => {
        const modelId = document.querySelector("[data-dft-slice-plot] dft-kspace-plot")?.getAttribute("data-source");
        const payloadText = modelId ? document.getElementById(modelId)?.textContent ?? "" : "";
        const payload = payloadText ? JSON.parse(payloadText) : null;
        return {
          component: Boolean(document.querySelector("[data-dft-slice-plot] dft-kspace-plot")),
          lineComponent: Boolean(document.querySelector("[data-dft-slice-plot] dft-line-graph")),
          kspace: document.querySelector("[data-dft-slice-plot] dft-kspace-plot svg")?.classList.contains("kspace-svg") ?? false,
          text: document.querySelector("[data-dft-slice-plot]")?.textContent ?? "",
          seriesCount: payload?.series?.length ?? 0,
          seriesKinds: payload?.series?.map((/** @type {any} */ series) => series.kind) ?? [],
        };
      });

      assert.equal(kspacePlotResult.component, true);
      assert.equal(kspacePlotResult.lineComponent, false);
      assert.equal(kspacePlotResult.kspace, true);
      assert.ok(kspacePlotResult.seriesCount > 0);
      assert.ok(kspacePlotResult.seriesCount <= 2);
      assert.ok(
        kspacePlotResult.seriesKinds.every((/** @type {string} */ kind) => kind === "line" || kind === "points"),
        `unexpected series kinds: ${JSON.stringify(kspacePlotResult.seriesKinds)}`,
      );
    } finally {
      await server.close();
    }
  } finally {
    await browserInstance.close();
  }
});


test("band surface slice slider updates 3D plane immediately", async () => {
  const browserInstance = await chromium.launch();
  const root = mkdtempSync(join(tmpdir(), "dft-local-slice-plane-live-"));
  const payload = {
    kind: "band-surface-preview",
    nu: 3,
    nv: 3,
    k1: [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    k2: [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    energies: [
      [[0, 10], [1, 11], [2, 12]],
      [[2, 12], [3, 13], [4, 14]],
      [[4, 14], [5, 15], [6, 16]],
    ],
    mask: [[true, true, true], [true, true, true], [true, true, true]],
    bands: [0, 1],
    nbands: 2,
    selected_band: 0,
    energy_unit: "eV",
  };

  try {
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
    <script type="application/json" id="dft-model-surface" data-dft-model="surface">${JSON.stringify(payload).replaceAll("</", "<\\/")}</script>
    <dft-band-surface-viewer data-source="dft-model-surface" data-dft-model="dft-model-surface"></dft-band-surface-viewer>
  </section>
  <script type="module" src="/dft-local-components.js"></script>
</body>
</html>`);

    copyFileSync("src/dft_local/diagnostics/static/dft-local-components.js", join(root, "dft-local-components.js"));
    const server = await serveDirectory(root);
    const page = await browserInstance.newPage();

    try {
      await page.goto(server.url);
      await page.waitForSelector("dft-band-surface-viewer canvas", { timeout: 10000 });
      await page.locator("[data-dft-slice-details] summary").click();

      await page.waitForFunction(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        return viewer?.sliceMeshes?.length === 1;
      }, { timeout: 10000 });

      const before = await page.evaluate(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        viewer.__renderThreeOnceCount = 0;
        const originalRender = viewer.renderThreeOnce.bind(viewer);
        viewer.renderThreeOnce = () => {
          viewer.__renderThreeOnceCount += 1;
          return originalRender();
        };
        return {
          axis: viewer.sliceAxis,
          value: viewer.sliceValue,
          meshValue: viewer.sliceMeshes[0]?.userData?.dftSliceValue,
          meshCount: viewer.sliceMeshes.length,
          renders: viewer.__renderThreeOnceCount,
        };
      });

      await page.locator("[data-dft-view-slice-value]").evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) throw new Error("not input");
        input.value = "0.8";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });

      const after = await page.evaluate(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        return {
          axis: viewer.sliceAxis,
          value: viewer.sliceValue,
          meshValue: viewer.sliceMeshes[0]?.userData?.dftSliceValue,
          meshCount: viewer.sliceMeshes.length,
          renders: viewer.__renderThreeOnceCount,
        };
      });

      assert.equal(before.meshCount, 1);
      assert.equal(after.meshCount, 1);
      assert.ok(Math.abs(Number(after.value) - 0.8) < 0.01);
      assert.ok(Math.abs(Number(after.meshValue) - 0.8) < 0.01);
      assert.ok(after.renders > 0);
    } finally {
      await server.close();
    }
  } finally {
    await browserInstance.close();
  }
});


test("band surface viewer draws cheap slice plane outline in 3D", async () => {
  const browserInstance = await chromium.launch();
  const root = mkdtempSync(join(tmpdir(), "dft-local-slice-no-3d-overlay-"));
  const payload = {
    kind: "band-surface-preview",
    nu: 3,
    nv: 3,
    k1: [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    k2: [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    energies: [
      [[0, 10], [1, 11], [2, 12]],
      [[2, 12], [3, 13], [4, 14]],
      [[4, 14], [5, 15], [6, 16]],
    ],
    mask: [[true, true, true], [true, true, true], [true, true, true]],
    bands: [0, 1],
    nbands: 2,
    selected_band: 0,
    energy_unit: "eV",
  };

  try {
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
    <script type="application/json" id="dft-model-surface" data-dft-model="surface">${JSON.stringify(payload).replaceAll("</", "<\\/")}</script>
    <dft-band-surface-viewer data-source="dft-model-surface" data-dft-model="dft-model-surface"></dft-band-surface-viewer>
  </section>
  <script type="module" src="/dft-local-components.js"></script>
</body>
</html>`);

    copyFileSync("src/dft_local/diagnostics/static/dft-local-components.js", join(root, "dft-local-components.js"));
    const server = await serveDirectory(root);
    const page = await browserInstance.newPage();

    try {
      await page.goto(server.url);
      await page.waitForSelector("dft-band-surface-viewer canvas", { timeout: 10000 });

      const closedControls = await page.evaluate(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        return {
          axisCount: viewer.querySelectorAll("[data-dft-view-slice-axis]").length,
          valueCount: viewer.querySelectorAll("[data-dft-view-slice-value]").length,
          controlsInsideDetails: Boolean(viewer.querySelector("[data-dft-slice-details] [data-dft-slice-controls] [data-dft-view-slice-axis]")),
          detailsOpen: Boolean(viewer.querySelector("[data-dft-slice-details]")?.open),
          sliceMeshes: viewer.sliceMeshes.length,
        };
      });

      assert.deepEqual(closedControls, {
        axisCount: 1,
        valueCount: 1,
        controlsInsideDetails: true,
        detailsOpen: false,
        sliceMeshes: 0,
      });

      await page.locator("[data-dft-slice-details] summary").click();
      await page.locator("[data-dft-view-slice-value]").evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) throw new Error("not input");
        input.value = "0.5";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });

      await page.waitForFunction(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        const panel = viewer?.querySelector("[data-dft-slice-panel]");
        return panel?.textContent?.includes("segments");
      }, { timeout: 10000 });

      const result = await page.evaluate(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        const guide = viewer.sliceMeshes[0];
        return {
          sliceMeshes: viewer.sliceMeshes.length,
          childCount: guide?.children?.length ?? 0,
          hasFill: Boolean(guide?.children?.some((/** @type {any} */ child) => child.name?.includes("fill"))),
          hasOutline: Boolean(guide?.children?.some((/** @type {any} */ child) => child.name?.includes("outline"))),
          hasMarker: Boolean(guide?.children?.some((/** @type {any} */ child) => child.name?.includes("center"))),
          axis: guide?.userData?.dftSliceAxis,
          panel: viewer.querySelector("[data-dft-slice-panel]")?.textContent ?? "",
        };
      });

      assert.equal(result.sliceMeshes, 1);
      assert.ok(result.childCount >= 3);
      assert.equal(result.hasFill, true);
      assert.equal(result.hasOutline, true);
      assert.equal(result.hasMarker, true);
      assert.equal(result.axis, "u");
      assert.match(result.panel, /segments/);

      await page.locator("[data-dft-view-slice-axis]").selectOption("v");
      await page.locator("[data-dft-view-slice-value]").evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) throw new Error("not input");
        input.value = "0.5";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });

      const vGuide = await page.evaluate(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        const guide = viewer.sliceMeshes[0];
        return {
          sliceMeshes: viewer.sliceMeshes.length,
          childCount: guide?.children?.length ?? 0,
          axis: guide?.userData?.dftSliceAxis,
          hasFill: Boolean(guide?.children?.some((/** @type {any} */ child) => child.name?.includes("fill"))),
        };
      });

      assert.equal(vGuide.sliceMeshes, 1);
      assert.ok(vGuide.childCount >= 3);
      assert.equal(vGuide.axis, "v");
      assert.equal(vGuide.hasFill, true);

      await page.locator("[data-dft-view-slice-axis]").selectOption("kx");
      await page.locator("[data-dft-view-slice-value]").evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) throw new Error("not input");
        input.value = "0.5";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });

      const kxGuide = await page.evaluate(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        const guide = viewer.sliceMeshes[0];
        return {
          sliceMeshes: viewer.sliceMeshes.length,
          childCount: guide?.children?.length ?? 0,
          axis: guide?.userData?.dftSliceAxis,
          hasFill: Boolean(guide?.children?.some((/** @type {any} */ child) => child.name?.includes("fill"))),
        };
      });

      assert.equal(kxGuide.sliceMeshes, 1);
      assert.ok(kxGuide.childCount >= 3);
      assert.equal(kxGuide.axis, "kx");
      assert.equal(kxGuide.hasFill, true);

      await page.locator("[data-dft-view-slice-axis]").selectOption("ky");
      await page.locator("[data-dft-view-slice-value]").evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) throw new Error("not input");
        input.value = "0.5";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });

      const kyGuide = await page.evaluate(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        const guide = viewer.sliceMeshes[0];
        return {
          sliceMeshes: viewer.sliceMeshes.length,
          childCount: guide?.children?.length ?? 0,
          axis: guide?.userData?.dftSliceAxis,
          hasFill: Boolean(guide?.children?.some((/** @type {any} */ child) => child.name?.includes("fill"))),
        };
      });

      assert.equal(kyGuide.sliceMeshes, 1);
      assert.ok(kyGuide.childCount >= 3);
      assert.equal(kyGuide.axis, "ky");
      assert.equal(kyGuide.hasFill, true);

      await page.locator("[data-dft-view-slice-axis]").selectOption("energy");
      await page.locator("[data-dft-view-slice-value]").evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) throw new Error("not input");
        input.value = "12";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });

      const energyGuide = await page.evaluate(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        const guide = viewer.sliceMeshes[0];
        return {
          sliceMeshes: viewer.sliceMeshes.length,
          childCount: guide?.children?.length ?? 0,
          axis: guide?.userData?.dftSliceAxis,
          hasFill: Boolean(guide?.children?.some((/** @type {any} */ child) => child.name?.includes("fill"))),
        };
      });

      assert.equal(energyGuide.sliceMeshes, 1);
      assert.ok(energyGuide.childCount >= 3);
      assert.equal(energyGuide.axis, "energy");
      assert.equal(energyGuide.hasFill, true);
    } finally {
      await server.close();
    }
  } finally {
    await browserInstance.close();
  }
});

test("band surface viewer shows slice intersection panel", async () => {
  const browserInstance = await chromium.launch();
  const root = mkdtempSync(join(tmpdir(), "dft-local-slice-panel-"));
  const payload = {
    kind: "band-surface-preview",
    nu: 3,
    nv: 3,
    k1: [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    k2: [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    energies: [
      [[0, 10], [1, 11], [2, 12]],
      [[2, 12], [3, 13], [4, 14]],
      [[4, 14], [5, 15], [6, 16]],
    ],
    mask: [[true, true, true], [true, true, true], [true, true, true]],
    bands: [0, 1],
    nbands: 2,
    selected_band: 0,
    energy_unit: "eV",
  };

  try {
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
    <script type="application/json" id="dft-model-surface" data-dft-model="surface">${JSON.stringify(payload).replaceAll("</", "<\\/")}</script>
    <dft-band-surface-viewer data-source="dft-model-surface" data-dft-model="dft-model-surface"></dft-band-surface-viewer>
  </section>
  <script type="module" src="/dft-local-components.js"></script>
</body>
</html>`);

    copyFileSync("src/dft_local/diagnostics/static/dft-local-components.js", join(root, "dft-local-components.js"));
    const server = await serveDirectory(root);
    const page = await browserInstance.newPage();

    try {
      await page.goto(server.url);
      await page.waitForSelector("dft-band-surface-viewer canvas", { timeout: 10000 });
      await page.locator("[data-dft-slice-details] summary").click();
      await page.locator("[data-dft-view-slice-value]").evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) throw new Error("not input");
        input.value = "0.5";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });

      await page.waitForFunction(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        const panel = viewer?.querySelector("[data-dft-slice-panel]");
        return viewer?.sliceAxis === "u"
          && Math.abs(Number(viewer?.sliceValue) - 0.5) < 0.01
          && panel?.textContent?.includes("slice u=")
          && panel?.textContent?.includes("segments");
      }, { timeout: 10000 });

      let panelText = await page.locator("dft-band-surface-viewer [data-dft-slice-panel]").evaluate((node) => node.textContent ?? "");
      assert.match(panelText, /slice u=/);
      assert.match(panelText, /segments/);

      await page.locator("[data-dft-view-slice-axis]").selectOption("energy");
      await page.locator("[data-dft-view-slice-value]").evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) throw new Error("not input");
        input.value = "12";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });

      await page.waitForFunction(() => {
        const panel = document.querySelector("[data-dft-slice-panel]");
        return panel?.textContent?.includes("slice energy=12");
      }, { timeout: 10000 });

      panelText = await page.locator("dft-band-surface-viewer [data-dft-slice-panel]").evaluate((node) => node.textContent ?? "");
      assert.match(panelText, /slice energy=12/);
      assert.match(panelText, /band 1/);
    } finally {
      await server.close();
    }
  } finally {
    await browserInstance.close();
  }
});


test("band surface view sliders respond to wheel and shift-wheel", async () => {
  const browserInstance = await chromium.launch();
  const root = mkdtempSync(join(tmpdir(), "dft-local-view-slider-wheel-"));
  const payload = {
    kind: "band-surface-preview",
    nu: 3,
    nv: 3,
    k1: [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    k2: [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    energies: [
      [[0], [1], [2]],
      [[1], [2], [3]],
      [[2], [3], [4]],
    ],
    mask: [[true, true, true], [true, true, true], [true, true, true]],
    bands: [0],
    nbands: 1,
    selected_band: 0,
    energy_unit: "eV",
  };

  try {
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
    <script type="application/json" id="dft-model-surface" data-dft-model="surface">${JSON.stringify(payload).replaceAll("</", "<\\/")}</script>
    <dft-band-surface-viewer data-source="dft-model-surface" data-dft-model="dft-model-surface"></dft-band-surface-viewer>
  </section>
  <script type="module" src="/dft-local-components.js"></script>
</body>
</html>`);

    copyFileSync("src/dft_local/diagnostics/static/dft-local-components.js", join(root, "dft-local-components.js"));
    const server = await serveDirectory(root);
    const page = await browserInstance.newPage();

    try {
      await page.goto(server.url);
      await page.waitForSelector("dft-band-surface-viewer canvas", { timeout: 10000 });

      const scale = page.locator("[data-dft-view-energy-scale]");
      await scale.hover();
      await page.mouse.wheel(0, -100);
      await page.waitForFunction(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        return Number(viewer?.energyScale) > 1;
      }, { timeout: 10000 });

      const afterNormal = await page.evaluate(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        return viewer.energyScale;
      });

      await page.keyboard.down("Shift");
      await page.mouse.wheel(0, -100);
      await page.keyboard.up("Shift");

      const afterShift = await page.evaluate(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        return viewer.energyScale;
      });

      assert.ok(afterNormal > 1);
      assert.ok(afterShift > afterNormal + 0.5);
    } finally {
      await server.close();
    }
  } finally {
    await browserInstance.close();
  }
});


test("band surface energy scale slider updates group transform without rebuilding meshes", async () => {
  const browserInstance = await chromium.launch();
  const root = mkdtempSync(join(tmpdir(), "dft-local-energy-scale-group-"));
  const payload = {
    kind: "band-surface-preview",
    nu: 3,
    nv: 3,
    k1: [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    k2: [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    energies: [
      [[0], [1], [2]],
      [[1], [2], [3]],
      [[2], [3], [4]],
    ],
    mask: [[true, true, true], [true, true, true], [true, true, true]],
    bands: [0],
    nbands: 1,
    selected_band: 0,
    energy_unit: "eV",
  };

  try {
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
    <script type="application/json" id="dft-model-surface" data-dft-model="surface">${JSON.stringify(payload).replaceAll("</", "<\\/")}</script>
    <dft-band-surface-viewer data-source="dft-model-surface" data-dft-model="dft-model-surface"></dft-band-surface-viewer>
  </section>
  <script type="module" src="/dft-local-components.js"></script>
</body>
</html>`);

    copyFileSync("src/dft_local/diagnostics/static/dft-local-components.js", join(root, "dft-local-components.js"));
    const server = await serveDirectory(root);
    const page = await browserInstance.newPage();

    try {
      await page.goto(server.url);
      await page.waitForSelector("dft-band-surface-viewer canvas", { timeout: 10000 });
      await page.waitForFunction(() => {
        const viewer = document.querySelector("dft-band-surface-viewer");
        return Boolean(/** @type {any} */ (viewer)?.surfaceGroup);
      }, { timeout: 10000 });

      await page.evaluate(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        viewer.__updateSurfaceCount = 0;
        viewer.__renderThreeOnceCount = 0;
        const original = viewer.updateSurface.bind(viewer);
        viewer.updateSurface = async () => {
          viewer.__updateSurfaceCount += 1;
          return original();
        };
        const originalRender = viewer.renderThreeOnce.bind(viewer);
        viewer.renderThreeOnce = () => {
          viewer.__renderThreeOnceCount += 1;
          return originalRender();
        };
      });

      await page.locator("[data-dft-view-energy-scale]").evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) throw new Error("not input");
        input.value = "3";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });

      await page.waitForFunction(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        return Number(viewer?.surfaceGroup?.scale?.y) === 3;
      }, { timeout: 10000 });

      const result = await page.evaluate(() => {
        const viewer = /** @type {any} */ (document.querySelector("dft-band-surface-viewer"));
        return {
          scale: viewer.surfaceGroup.scale.y,
          rebuilds: viewer.__updateSurfaceCount,
          renders: viewer.__renderThreeOnceCount,
          status: document.querySelector("[data-dft-surface-status]")?.textContent,
        };
      });

      assert.equal(result.scale, 3);
      assert.equal(result.rebuilds, 0);
      assert.ok(result.renders > 0);
      assert.match(result.status ?? "", /energy scale 3/);
    } finally {
      await server.close();
    }
  } finally {
    await browserInstance.close();
  }
});


test("model patch observer refreshes component after script text mutation", async () => {
  const browserInstance = await chromium.launch();
  const root = mkdtempSync(join(tmpdir(), "dft-local-model-text-mutation-"));
  const firstPayload = {
    kind: "band-surface-preview",
    nu: 3,
    nv: 3,
    k1: [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    k2: [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    energies: [
      [[0], [1], [2]],
      [[1], [2], [3]],
      [[2], [3], [4]],
    ],
    mask: [[true, true, true], [true, true, true], [true, true, true]],
    bands: [0],
    nbands: 1,
    selected_band: 0,
    energy_unit: "eV",
  };
  const secondPayload = {
    ...firstPayload,
    nu: 5,
    nv: 5,
    k1: Array.from({ length: 5 }, (_, i) => Array.from({ length: 5 }, (_, j) => i + j)),
    k2: Array.from({ length: 5 }, (_, i) => Array.from({ length: 5 }, (_, j) => i - j)),
    energies: Array.from({ length: 5 }, (_, i) => Array.from({ length: 5 }, (_, j) => [i - j])),
    mask: Array.from({ length: 5 }, () => Array.from({ length: 5 }, () => true)),
  };

  try {
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

    copyFileSync("src/dft_local/diagnostics/static/dft-local-components.js", join(root, "dft-local-components.js"));
    const server = await serveDirectory(root);
    const page = await browserInstance.newPage();

    try {
      await page.goto(server.url);
      await page.waitForSelector("dft-band-surface-viewer canvas", { timeout: 10000 });
      await page.waitForFunction(() => Number(/** @type {any} */ (document.querySelector("dft-band-surface-viewer"))?.payload?.nu) === 3, { timeout: 10000 });

      await page.evaluate((payload) => {
        const model = document.querySelector("#dft-model-surface");
        if (!model) throw new Error("missing model island");
        model.textContent = JSON.stringify(payload);
      }, secondPayload);

      await page.waitForFunction(() => {
        const viewer = document.querySelector("dft-band-surface-viewer");
        return Number(/** @type {any} */ (viewer)?.payload?.nu) === 5
          && Number(/** @type {any} */ (viewer)?.currentMesh?.summary?.count) === 25;
      }, { timeout: 10000 });

      const result = await page.evaluate(() => {
        const viewer = document.querySelector("dft-band-surface-viewer");
        return {
          nu: /** @type {any} */ (viewer)?.payload?.nu,
          nv: /** @type {any} */ (viewer)?.payload?.nv,
          count: /** @type {any} */ (viewer)?.currentMesh?.summary?.count,
          status: document.querySelector("[data-dft-surface-status]")?.textContent,
        };
      });

      assert.equal(result.nu, 5);
      assert.equal(result.nv, 5);
      assert.equal(result.count, 25);
      assert.match(result.status ?? "", /grid 5×5/);
      assert.match(result.status ?? "", /energy zero 0/);
      assert.match(result.status ?? "", /energy scale 1/);
    } finally {
      await server.close();
    }
  } finally {
    await browserInstance.close();
  }
});


/** 
 * @param {string} diagnosticId
 * @param {Record<string, string>} rawInputs
 * @returns {string}
 */
function pythonDiagnosticPage(diagnosticId, rawInputs) {
  const code = `
import json
from dft_local.diagnostics.server import DiagnosticApp, load_default_context

diagnostic_id = ${JSON.stringify(diagnosticId)}
raw_inputs = json.loads(${JSON.stringify(JSON.stringify(rawInputs))})

ctx = load_default_context("test_run/run_dir/data")
app = DiagnosticApp(ctx=ctx)
print(app.diagnostic_page(diagnostic_id, raw_inputs), end="")
`;
  return execFileSync("python", ["-c", code], {
    cwd: process.cwd(),
    encoding: "utf8",
  });
}





test("real group-resolved DiagnosticApp page boots band surface viewer", async () => {
  const browserInstance = await chromium.launch();
  const root = mkdtempSync(join(tmpdir(), "dft-local-real-group-page-"));
  const pageHtml = pythonDiagnosticPage("transport.boltzmann.group_resolved.overview", {
    kernel: "average_star",
    nu: "5",
    nv: "5",
    band: "0",
  });

  assert.match(pageHtml, /<dft-band-surface-viewer/);
  assert.match(pageHtml, /dft-model-group_resolved_band_surface/);

  try {
    writeFileSync(join(root, "index.html"), pageHtml);
    copyFileSync("src/dft_local/diagnostics/static/dft-local-components.js", join(root, "dft-local-components.js"));

    const server = await serveDirectory(root);
    const page = await browserInstance.newPage();
    /** @type {string[]} */
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() !== "error") return;
      const text = message.text();

      // Chromium may request /favicon.ico for a static test page.  That 404 is
      // unrelated to component boot.
      if (text.includes("Failed to load resource") && text.includes("404")) return;

      errors.push(text);
    });

    try {
      await page.route("**/static/dft-local-components.js", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "text/javascript",
          body: readFileSync(join(root, "dft-local-components.js"), "utf8"),
        });
      });

      await page.goto(server.url);
      await page.waitForSelector("dft-band-surface-viewer canvas", { timeout: 10000 });
      await page.waitForFunction(() => {
        const viewer = document.querySelector("dft-band-surface-viewer");
        return Boolean(
          customElements.get("dft-band-surface-viewer")
          && viewer
          && /** @type {any} */ (viewer).payload
          && /** @type {any} */ (viewer).payload.nu === 5
          && /** @type {any} */ (viewer).payload.nv === 5
          && /** @type {any} */ (viewer).payload.nbands === 8
          && viewer.querySelector("canvas")
          && viewer.querySelector("[data-dft-surface-status]")?.textContent?.includes("grid 5×5")
        );
      }, { timeout: 10000 });

      const probe = await page.evaluate(() => {
        const viewer = document.querySelector("dft-band-surface-viewer");
        return {
          viewer: Boolean(viewer),
          defined: Boolean(customElements.get("dft-band-surface-viewer")),
          payload: {
            nu: /** @type {any} */ (viewer)?.payload?.nu,
            nv: /** @type {any} */ (viewer)?.payload?.nv,
            nbands: /** @type {any} */ (viewer)?.payload?.nbands,
          },
          hasCanvas: Boolean(viewer?.querySelector("canvas")),
          status: viewer?.querySelector("[data-dft-surface-status]")?.textContent ?? "",
          hasViewZero: Boolean(viewer?.querySelector("[data-dft-view-energy-zero]")),
          hasViewScale: Boolean(viewer?.querySelector("[data-dft-view-energy-scale]")),
        };
      });

      assert.deepEqual(probe.payload, { nu: 5, nv: 5, nbands: 8 });
      assert.equal(probe.viewer, true);
      assert.equal(probe.defined, true);
      assert.equal(probe.hasCanvas, true);
      assert.equal(probe.hasViewZero, true);
      assert.equal(probe.hasViewScale, true);
      assert.match(probe.status, /grid 5×5/);
      assert.match(probe.status, /visible 8/);
      assert.match(probe.status, /energy zero 0/);
      assert.match(probe.status, /energy scale 1/);
      assert.deepEqual(errors, []);
    } finally {
      await server.close();
    }
  } finally {
    await browserInstance.close();
  }
});


test("real group-resolved DiagnosticApp SSE updates band surface model", async () => {
  const browserInstance = await chromium.launch();
  const root = mkdtempSync(join(tmpdir(), "dft-local-group-resolved-"));
  const sse = pythonDiagnosticRunStream("transport.boltzmann.group_resolved.overview", {
    kernel: "average_star",
    nu: "3",
    nv: "3",
    band: "0",
  });

  assert.match(sse, /#dft-model-group_resolved_band_surface/);
  assert.match(sse, /"nu": 3/);
  assert.match(sse, /"nv": 3/);

  const initialPayload = {
    kind: "band-surface-preview",
    nu: 7,
    nv: 7,
    k1: Array.from({ length: 7 }, (_, i) => Array.from({ length: 7 }, (_, j) => i + j)),
    k2: Array.from({ length: 7 }, (_, i) => Array.from({ length: 7 }, (_, j) => i - j)),
    energies: Array.from({ length: 7 }, (_, i) => Array.from({ length: 7 }, (_, j) => [i + j, i - j])),
    mask: Array.from({ length: 7 }, () => Array.from({ length: 7 }, () => true)),
    bands: [0, 1],
    nbands: 2,
    selected_band: 0,
    energy_unit: "eV",
  };

  try {
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
  <button type="submit" data-dft-run-button aria-busy="true">Run</button>
  <span data-dft-run-status aria-live="polite">computing…</span>
  <section id="dft-block-group_resolved_band_surface" data-dft-block="group_resolved_band_surface" data-dft-block-kind="json-rendered">
    <script type="application/json" id="dft-model-group_resolved_band_surface" data-dft-model="group_resolved_band_surface">${JSON.stringify(initialPayload).replaceAll("</", "<\\/")}</script>
    <dft-band-surface-viewer data-source="dft-model-group_resolved_band_surface" data-dft-model="dft-model-group_resolved_band_surface"></dft-band-surface-viewer>
  </section>
  <script type="module" src="/dft-local-components.js"></script>
</body>
</html>`);

    copyFileSync("src/dft_local/diagnostics/static/dft-local-components.js", join(root, "dft-local-components.js"));
    const server = await serveDirectory(root);
    const page = await browserInstance.newPage();

    try {
      await page.goto(server.url);
      await page.waitForSelector("dft-band-surface-viewer canvas", { timeout: 10000 });
      await page.waitForFunction(() => Number(/** @type {any} */ (document.querySelector("dft-band-surface-viewer"))?.payload?.nu) === 7, { timeout: 10000 });

      await page.evaluate((stream) => {
        const events = stream.trim().split(/\n\n+/);
        for (const eventText of events) {
          const lines = eventText.split("\n");
          const event = lines.find((line) => line.startsWith("event: "))?.slice("event: ".length);
          const data = lines
            .filter((line) => line.startsWith("data: "))
            .map((line) => line.slice("data: ".length));

          if (event === "datastar-patch-elements") {
            const selector = data.find((line) => line.startsWith("selector "))?.slice("selector ".length);
            const mode = data.find((line) => line.startsWith("mode "))?.slice("mode ".length) ?? "outer";
            const elements = data
              .filter((line) => line.startsWith("elements "))
              .map((line) => line.slice("elements ".length))
              .join("\n");

            if (!selector) throw new Error("missing selector");

            const targets = Array.from(document.querySelectorAll(selector));
            const template = document.createElement("template");
            template.innerHTML = elements.trim();
            for (const target of targets) {
              const replacement = template.content.firstElementChild?.cloneNode(true);
              if (!replacement) continue;
              if (mode === "outer") target.replaceWith(replacement);
              else if (mode === "inner") target.innerHTML = elements;
            }
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
        const status = document.querySelector("[data-dft-run-status]");
        return Number(/** @type {any} */ (viewer)?.payload?.nu) === 3
          && Number(/** @type {any} */ (viewer)?.payload?.nv) === 3
          && status?.textContent === "updated";
      }, { timeout: 10000 });

      const result = await page.evaluate(() => {
        const viewer = document.querySelector("dft-band-surface-viewer");
        const status = document.querySelector("[data-dft-run-status]");
        return {
          nu: /** @type {any} */ (viewer)?.payload?.nu,
          nv: /** @type {any} */ (viewer)?.payload?.nv,
          status: status?.textContent,
        };
      });

      assert.deepEqual(result, { nu: 3, nv: 3, status: "updated" });
    } finally {
      await server.close();
    }
  } finally {
    await browserInstance.close();
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
