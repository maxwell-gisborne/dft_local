import test from "node:test";
import assert from "node:assert/strict";

import { nice } from "./dft-local-components.js";

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
