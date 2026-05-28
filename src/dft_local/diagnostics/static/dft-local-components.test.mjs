import test from "node:test";
import assert from "node:assert/strict";

import { nice, graphBounds, zoomView, panView } from "./dft-local-components.js";

test("nice formats ordinary numbers", () => {
  assert.equal(nice(0), "0");
  assert.equal(nice(1.25), "1.25");
});

test("nice formats very small and very large numbers exponentially", () => {
  assert.equal(nice(1e-4), "1.00e-4");
  assert.equal(nice(1e5), "1.00e+5");
});

test("nice handles non-finite values", () => {
  assert.equal(nice(Number.NaN), "");
  assert.equal(nice(Number.POSITIVE_INFINITY), "");
});


test("graphBounds returns padded bounds", () => {
  const bounds = graphBounds({
    id: "g",
    title: "G",
    x_label: "x",
    y_label: "y",
    series: [
      {
        name: "s",
        kind: "line",
        points: [
          { x: 0, y: 0 },
          { x: 10, y: 20 },
        ],
      },
    ],
  });

  assert.equal(bounds.xmin, 0);
  assert.equal(bounds.xmax, 10);
  assert.equal(bounds.ymin, -1.2);
  assert.equal(bounds.ymax, 21.2);
});

test("zoomView contracts around fractional point", () => {
  const view = { xmin: 0, xmax: 10, ymin: 0, ymax: 10 };
  const got = zoomView(view, 0.5, 0.5, 0.5);

  assert.deepEqual(got, { xmin: 2.5, xmax: 7.5, ymin: 2.5, ymax: 7.5 });
});

test("panView shifts the visible window", () => {
  const view = { xmin: 0, xmax: 10, ymin: 0, ymax: 20 };
  const got = panView(view, 0.1, -0.25);

  assert.deepEqual(got, { xmin: -1, xmax: 9, ymin: -5, ymax: 15 });
});
