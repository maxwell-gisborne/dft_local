# dft_local Consolidated Handoff

This handoff consolidates the current project state, recent diagnostic work, maintenance rules, and next steps for the `dft_local` repository. It is intended to replace the scattered handoff notes and make it easier for the next person to continue without rediscovering the same architectural constraints and debugging lessons.

> Last known committed checkpoint from the conversation: `c622313 Add Gaussian DOS diagnostic viewer`
>
> Note: this document is based on the supplied handoff notes and live debugging transcript. The repository itself was not mounted in the assistant sandbox, so commit history beyond the user-provided logs was not independently inspected here.

---

## 1. Project purpose and mental model

`dft_local` is a diagnostics and post-processing project for DFT-derived graphene Hamiltonian/overlap data. It connects local DFT matrix data to group-labelled kernels, group-Fourier symbols, band surfaces, velocities, and Boltzmann conductivity diagnostics.

There are two overlapping pipelines.

### 1.1 Scientific/numerical pipeline

```text
BigDFT / dft_local data
  -> group-labelled local orbital blocks
  -> kernels K_H(h), K_S(h)
  -> group-Fourier symbols Ĥ(Ω), Ŝ(Ω)
  -> generalized eigenproblem
  -> energy sheets / velocities
  -> Boltzmann conductivity
  -> diagnostic panels and plots
```

### 1.2 Diagnostics UI pipeline

```text
diagnostic context
  -> diagnostic specs
  -> computed diagnostic results
  -> server-rendered HTML + JSON model islands
  -> interactive browser diagnostics
  -> Datastar model patches on rerun
```

Central UI rule:

```text
server owns data
browser owns view state
Datastar swaps model islands / small blocks
components preserve camera, selection, zoom, hidden bands, and local UI state
```

Avoid replacing whole interactive components when only the diagnostic data changes. Patch the JSON model island and call the component update path.

---

## 2. Important repository areas

### 2.1 Diagnostics framework

```text
src/dft_local/diagnostics/server.py
src/dft_local/diagnostics/render.py
src/dft_local/diagnostics/context.py
src/dft_local/diagnostics/discovery.py
src/dft_local/diagnostics/static/dft-local-components.js
src/dft_local/diagnostics/static/dft-local-components.test.mjs
src/dft_local/diagnostics/static/dft-local-components.browser.test.mjs
```

Main routes/classes:

```text
DiagnosticApp
DiagnosticASGI

/
/docs
/docs/<doc_id>
/static/<path>
/d/<diagnostic_id>
/d-run/<diagnostic_id>
```

`/d-run/<diagnostic_id>` returns Datastar SSE patches.

Important server parsing rule:

```python
inputs = parse_inputs(spec, raw_inputs)
```

Do not use:

```python
inputs = parse_inputs(spec.inputs, raw_inputs)
```

### 2.2 Core matrix/kernel layer

```text
src/dft_local/core/
  dataset.py
  sparse.py
  kernels.py
  local_problem.py
  numerics.py
  units.py
```

Important objects/path:

```text
GdKernelArrays
  h_m
  h_n
  h_eps
  blocks

GdKernelArrays.symbol_generic(k1, k2)
GdKernelArrays.symbol_fixed(k1, k2, sigma)
SymbolPair(KH, KS, k1, k2, degree=2, sigma=None).form()
LocalProblem.solve()
```

Production rule:

```text
Do not recreate symbol logic in parallel.
Boltzmann, bands, validation, and diagnostics should reuse GdKernelArrays.symbol_* and SymbolPair.form().
```

### 2.3 Band and group-resolved diagnostics

```text
src/dft_local/transport/bands/
  core.py
  diagnostics.py
  energy_surface_rectification.py

src/dft_local/transport/boltzmann/group_resolved/
  diagnostics.py
```

### 2.4 Boltzmann calculation and validation

```text
src/dft_local/transport/boltzmann/calculation/
  core.py
  diagnostics.py
  test_core.py
  test_conductivity_business_logic.py

src/dft_local/transport/boltzmann/validation/
  core.py
  diagnostics.py
  docs.md
  test_diagnostics.py
```

Important derivative helpers:

```python
gd_symbol_derivative_generic(...)
gd_symbol_derivative_fixed(...)
gd_symbol_derivatives(...)
```

### 2.5 Ashcroft/Vincent comparison

```text
src/dft_local/transport/boltzmann/ashcroft_comparison/
  core.py
  diagnostics.py
  docs.md
  test_diagnostics.py
  epsilon_of_k.txt
  vincents_results.md
  ai.txt
```

---

## 3. Diagnostics rendering contract

Rendered diagnostic blocks should use the block shell:

```html
<section
  id="dft-block-<block-id>"
  data-dft-block="<block-id>"
  data-dft-block-kind="<kind>">
  ...
</section>
```

Known block kinds:

```text
static-html
stateful-html
json-rendered
```

JSON-rendered components use model islands:

```html
<script type="application/json"
        id="dft-model-band_region_surface"
        data-dft-model="band_region_surface">
  {...payload...}
</script>
<dft-band-surface-viewer
  data-source="dft-model-band_region_surface"
  data-dft-model="dft-model-band_region_surface">
</dft-band-surface-viewer>
```

The model refresh protocol is:

```js
function refreshDftModels(root = document) {
  for (const element of Array.from(root.querySelectorAll("[data-dft-model]"))) {
    ...
    if (typeof maybeUpdater.updateModel === "function") {
      maybeUpdater.updateModel(model);
    }
  }
}
```

Any model-backed custom element must implement:

```js
updateModel(model) {
  ...
}
```

`refreshModel()` alone is not enough.

### 3.1 Datastar patch rules

For `json-rendered` blocks:

```text
patch only JSON model scripts
then execute window.dftRefreshModels?.(document)
```

For `stateful-html` table blocks:

```text
capture table state once
patch table block HTML
restore table state once
```

For `static-html` blocks:

```text
patch whole block shell
```

Do not replace the whole diagnostic result section if the page contains viewers, tables, selected rows, camera state, graph zoom/pan, or selected k-points.

---

## 4. Browser components and state

Important custom elements:

```text
dft-line-graph
dft-kspace-plot
dft-band-surface-viewer
dft-band-readout
dft-kpoint-readout
dft-band-controls
dft-dos-idos-viewer
```

Most stateful components:

```text
dft-band-surface-viewer
dft-line-graph
dft-kspace-plot
```

The browser helper functions exposed on `window` include:

```js
captureDftTableState(root)
restoreDftTableState(state, root)
preserveDftTableState(replace, root)
```

Tables are not web components. They are plain HTML tables with state restoration:

```html
<table data-dft-table="...">
<tr data-dft-row-id="...">
```

Table state includes:

```text
selected row ids
hovered row id
scroll position
```

---

## 5. Band-surface viewer current state

The band-surface viewer currently has:

```text
one set of viewer-owned controls
domain dropdown
visible intersection plot
visible k-space slice guide
Shift + wheel dolly zoom
corrected graphene BZ/K geometry
state-preserving Datastar model patch path
on-demand Three rendering
```

Recent viewer-related checkpoint before the DOS work:

```text
2f4325d Correct graphene Brillouin zone vertices
```

Recent viewer commits in order from the handoff:

```text
8a794a5 Make band surface slice plot visible
b8c8f2c Add band surface domain selector
2948498 Add band surface dolly zoom control
3142af1 Refine reciprocal reference plane styling
0f27e1f Show bold reciprocal reference geometry
474ef6f Move slice controls into intersection panel
cecf6e4 Add visible k-space slice guides
2f4325d Correct graphene Brillouin zone vertices
```

Checks were green after BZ correction:

```text
npm run js:check
  86 JS unit tests passed
  23 browser tests passed

focused Python diagnostics:
  51 passed
```

### 5.1 On-demand rendering rule

Do not bring back a permanent render loop.

Scene mutations should explicitly call:

```js
renderThreeOnce()
```

Important mutations include:

```text
domain dropdown changes
energy scale / zero
band visibility
slice overlay
camera dolly zoom
model update
```

Camera drag should stay cheap:

```text
camera drag:
  may move camera
  may render
  must not recompute slices
  must not rebuild surface
```

Debugging stale viewer visuals should follow:

```text
1. check state changed
2. check Three object changed
3. check renderThreeOnce was called
4. only then decide whether it is a visual clarity problem
```

Likely methods to audit when visuals go stale:

```text
applyEnergyTransform()
applyBandVisibility()
updateSelectedMarker()
updateSliceOverlay()
updateSurface()
```

### 5.2 Domain selector

The viewer now has one k-domain dropdown:

```text
primitive cell
BZ hexagon
extended hexagon
```

Meanings:

```text
primitive cell:
  origin-rooted reciprocal primitive parallelogram
  0, b1, b1+b2, b2

BZ hexagon:
  central graphene first Brillouin zone
  K/K′ vertices

extended hexagon:
  larger reciprocal-lattice shell through nearest reciprocal lattice points
```

Compatibility:

```text
bandSurfaceMeshDataWithDomain(...)
  new surface-domain helper

bandSurfaceMeshDataWithMask(...)
  kept as wrapper for old tests/call sites
```

One slice helper still accepts old `{ useMask }` options. Current bridge:

```js
{ useMask: this.domainMode === "bz" }
```

Long-term cleanup could make slice segmentation domain-aware too.

### 5.3 Three different hexagons

Do not confuse:

```text
primitive cell:
  reciprocal primitive parallelogram

BZ hexagon:
  first Brillouin zone / Wigner-Seitz cell
  vertices are K/K′ points

extended reciprocal hexagon:
  larger shell through nearest reciprocal lattice points
  not the BZ
```

Past confusion came from naming the large reciprocal shell “hexagon” and drawing a plausible but wrong π-sized BZ.

---

## 6. Correct reciprocal reference geometry

The old central BZ fallback hexagon used:

```text
(π,0), (π,π), (0,π), (-π,0), (-π,-π), (0,-π)
```

This was wrong for the current coordinate convention.

The viewer coordinate map is:

```js
bandBasisToCartesian(k1, k2) = {
  x: k1 - 0.5 * k2,
  y: sqrt(3) / 2 * k2,
}
```

So the display metric is:

```text
|k|² = k1² - k1 k2 + k2²
```

In this convention, one reciprocal primitive period is `2π` in each coordinate. The graphene first BZ corners, i.e. K/K′ points, are:

```text
( 4π/3,  2π/3)
( 2π/3,  4π/3)
(-2π/3,  2π/3)
(-4π/3, -2π/3)
(-2π/3, -4π/3)
( 2π/3, -2π/3)
```

This was patched in:

```text
src/dft_local/transport/bands/core.py
  bz_hexagon_vertices()

src/dft_local/diagnostics/static/dft-local-components.js
  threeHexagonReferenceData fallback

src/dft_local/transport/bands/diagnostics.py
  synthetic surface payload fallback
```

Important source-of-truth rule:

```text
server payloads often include payload.bz_hexagon
payload.bz_hexagon overrides JS fallback
therefore fixing JS alone is not enough
Python bz_hexagon_vertices() is the real source of truth
```

### 6.1 Dirac/K diagnostic lesson

When checking whether Dirac points sit at K, do not start with a minimum-gap plot. Better order:

```text
1. inspect coordinate map bandBasisToCartesian()
2. derive analytic reciprocal-space K points
3. compare drawn BZ/K markers to those points
4. only then inspect band degeneracies
```

If Dirac points still do not sit at K after the BZ correction, inspect:

```text
Are payload k1,k2 coordinates actually reciprocal primitive phase coordinates?
Or are they group/irrep coordinates from G_d or G_s with an extra linear transform?
```

Recommended next overlay:

```text
analytic K markers from bz_hexagon_vertices()
actual sample points nearest those K markers
readout of nearest grid indices and k values
```

---

## 7. DOS/IDOS diagnostic: latest committed feature

Latest known commit:

```text
c622313 Add Gaussian DOS diagnostic viewer
```

Commit summary:

```text
3 files changed
135 insertions
51 deletions
```

Working files:

```text
src/dft_local/transport/boltzmann/group_resolved/diagnostics.py
src/dft_local/diagnostics/static/dft-local-components.js
src/dft_local/diagnostics/static/dft-local-components.test.mjs
```

### 7.1 Purpose

The DOS/IDOS diagnostic plot was changed from a histogram to a Gaussian-broadened DOS curve, backed by the same JSON model island/Datastar refresh system as the other interactive diagnostics.

Original request:

```text
Convolve with a Gaussian instead of doing a histogram, where sigma follows the same logic as bin_count.
```

The intended model:

```text
DOS(E_i) = sum_(k,n) w_k exp(-(E_i - E_(k,n))^2 / (2 sigma^2)) / (sqrt(2 pi) sigma)
```

Important distinction:

```text
dos_resolution_count controls smoothing width
dos_plot_sample_count controls how many x-points draw the curve
```

The old histogram had 96 bins. With Gaussian broadening, 96 should not be both the smoothing resolution and the number of displayed vertices, because that still looks jagged. The display grid is denser.

### 7.2 Python-side intended state

Expected constants/logic:

```python
dos_resolution_count = 96
dos_plot_sample_count = 8 * dos_resolution_count

energy_values = energy_flat.ravel()
state_weights = np.broadcast_to(
    (spin_degeneracy * normalised_k_weights)[:, None],
    energy_flat.shape,
).ravel()

energy_min = float(np.min(energy_values))
energy_max = float(np.max(energy_values))
energy_range = max(energy_max - energy_min, np.finfo(float).eps)

dos_sigma = energy_range / float(dos_resolution_count)
dos_grid = np.linspace(
    energy_min - 3.0 * dos_sigma,
    energy_max + 3.0 * dos_sigma,
    dos_plot_sample_count,
)

gaussian_arg = (dos_grid[:, None] - energy_values[None, :]) / dos_sigma
gaussian_kernel = np.exp(-0.5 * gaussian_arg * gaussian_arg) / (
    np.sqrt(2.0 * np.pi) * dos_sigma
)
dos_density = gaussian_kernel @ state_weights

if dos_grid.size > 1:
    dE = float(dos_grid[1] - dos_grid[0])
else:
    dE = energy_range
idos_counts = np.cumsum(dos_density) * dE
dos_centres = dos_grid
```

Payload metadata should include:

```python
"sample_count": int(energy_flat.size),
"kpoint_count": int(energy_flat.shape[0]),
"band_count": int(energy_flat.shape[1]),
"dos_resolution_count": int(dos_resolution_count),
"dos_plot_sample_count": int(dos_plot_sample_count),
"dos_sigma": float(dos_sigma),
```

### 7.3 DOS source audit

Use this audit:

```bash
python - <<'PY'
from pathlib import Path

s = Path("src/dft_local/transport/boltzmann/group_resolved/diagnostics.py").read_text()
print("dos_payload count:", s.count("dos_payload = {"))
print("np.histogram count:", s.count("np.histogram"))
PY
```

Expected:

```text
dos_payload count: 1
np.histogram count: 0
```

This was green before the commit.

### 7.4 JS custom element

New element:

```js
class DftDosIdosViewer extends HTMLElement {
  connectedCallback() {
    this.renderFromModel();
  }

  refreshModel() {
    this.renderFromModel();
  }

  /**
   * @param {Record<string, unknown> | null | undefined} model
   */
  updateModel(model) {
    this.renderFromPayload(model);
  }

  modelPayload() {
    const sourceId = this.getAttribute("data-dft-model") || this.getAttribute("data-source");
    ...
  }

  renderFromModel() {
    this.renderFromPayload(this.modelPayload());
  }

  /**
   * @param {Record<string, unknown> | null | undefined} payload
   */
  renderFromPayload(payload) {
    ...
  }
}
```

Most important method:

```js
updateModel(model) {
  this.renderFromPayload(model);
}
```

The existing refresh system calls `updateModel(model)`, not only `refreshModel()`.

Registration:

```js
if (!customElements.get("dft-dos-idos-viewer")) {
  customElements.define("dft-dos-idos-viewer", DftDosIdosViewer);
}
```

### 7.5 DOS/IDOS debugging lessons

The JS file was corrupted during patch attempts. Three layers occurred:

1. Literal newline corruption:

```text
physical lines: 1
literal backslash-n count: 5415
```

This made Node treat nearly the whole file as a single `//` comment. The module imported with no exports, producing errors like:

```text
SyntaxError: The requested module './dft-local-components.js' does not provide an export named 'allBandIndices'
```

2. Global newline repair broke intentional JS string literals:

```js
.join("\n")
```

became illegal multiline strings like:

```js
.join("
")
```

3. A generated export block was accidentally duplicated or inserted in the wrong place. The observed tail included:

```js
export {

// Test-facing helper exports.
export {
```

The duplicate `export {` was removed. The export block is top-level.

Do not blindly patch exports. The `allBandIndices` export error was a symptom of a commented-out/corrupted file, not simply a missing export.

### 7.6 DOS/IDOS checks passed before commit

Final checks before the commit:

```text
JS syntax: passed
full JS check: passed
DOS audit:
  dos_payload count: 1
  np.histogram count: 0
focused Python:
  25 passed
```

Manual smoke test still useful:

```text
group-resolved diagnostic
change nu/nv
Run
confirm DOS/IDOS curve updates from model island
confirm curve samples = 768
confirm sigma finite
confirm no stale histogram-style bars
```

---

## 8. Unit, quantity, table display, and JSON copy system

Global diagnostic rule:

```text
Any semantic numeric diagnostic value in a table should be wrapped as a typed DisplayQuantity,
not manually formatted with an f-string.
```

Core files:

```text
src/dft_local/core/units.py
src/dft_local/core/dataset.py
src/dft_local/diagnostics/render.py
src/dft_local/transport/boltzmann/calculation/core.py
src/dft_local/transport/boltzmann/calculation/diagnostics.py
src/dft_local/transport/boltzmann/ashcroft_comparison/diagnostics.py
```

Recent relevant commits:

```text
e4c651c  Unwrap typed diagnostics in table JSON copy
e7eb593  Format diagnostic quantities compactly
1a27b1a  Pretty-print diagnostic scientific notation
```

### 8.1 Dimensions

A `Dimension` represents physical kind, not just units. Examples:

```python
DIMENSIONLESS
ENERGY
LENGTH
TIME
TEMPERATURE
VELOCITY
WAVEVECTOR
CONDUCTIVITY
KSPACE_AREA
```

Dimensionless values should still be wrapped if they are semantic data:

```python
DIMENSIONLESS
```

Do not represent semantic dimensionless numbers as strings like:

```python
"1.14528992e+00"
```

### 8.2 Units

A `Unit` represents display/conversion for a dimension.

Example:

```python
SIEMENS_PER_METER = Unit(
    symbol="S/m",
    dimension=CONDUCTIVITY,
    scale_to_si=1.0,
)
```

Dimensionless unit:

```python
UNITLESS = Unit(
    symbol="",
    dimension=DIMENSIONLESS,
    scale_to_si=1.0,
)
```

The empty symbol is deliberate. Do not use symbol `"1"` because it clutters tables.

Percent:

```python
PERCENT = Unit(
    symbol="%",
    dimension=DIMENSIONLESS,
    scale_to_si=0.01,
)
```

Current convention:

```text
DisplayQuantity.value stores the displayed percent number, not the fraction.
_dq_percent(10.675, ...) displays 10.675 % and JSON copies 10.675.
```

### 8.3 DisplayQuantity

Conceptual form:

```python
DisplayQuantity(
    value=<float>,
    dimension=<Dimension>,
    unit=<Unit>,
    name=<str>,
)
```

Roles:

```text
HTML display:
  central renderer prints compact value and unit suffix

JSON copy/export:
  unwraps raw numeric value
```

Do not pre-format numeric diagnostic data at the call site.

### 8.4 Formatting policy

Current display formatting:

```text
precision: 6 significant figures
abs(x) < 1e-3 -> scientific notation
abs(x) >= 1e4 -> scientific notation
otherwise -> compact ordinary notation
```

Examples:

```text
0.142682310      -> 0.142682
0.00361688594    -> 0.00361689
0.000361688594   -> 3.61689 × 10^-4
24375.6681       -> 2.43757 × 10^4
1.14528992       -> 1.14529
1.0              -> 1
```

Only display formatting changes. Stored/copy JSON remains raw.

### 8.5 JSON copy rule

```text
DisplayQuantity -> raw JSON number
plain string     -> string
plain float      -> old display-formatted string unless deliberately wrapped
```

This is deliberate: JSON export should not guess whether a string is data or presentation.

### 8.6 Ashcroft diagnostic quantity helpers

The Ashcroft diagnostics domain currently defines helpers:

```python
_dq_unitless
_dq_percent
_dq_wavevector
_dq_conductivity
_dq_velocity
_dq_electric_field
_dq_raw_velocity_weight
_dq_length
_dq_temperature
```

Local units/dimensions include:

```python
SIEMENS_PER_METER
METER_PER_SECOND
METER
PER_METER
UNITLESS
PERCENT
VOLT_PER_METER
RAW_VELOCITY_WEIGHT_UNIT
ELECTRIC_FIELD_STRENGTH = Dimension((1, 1, -3, -1, 0))
```

Potential future cleanup:

```text
Move ELECTRIC_FIELD_STRENGTH and electric-field units into core.units if needed outside Ashcroft.
Possibly move PERCENT and common display helpers into shared diagnostics utilities.
```

### 8.7 Audit for manual f-string numeric cells

Use:

```bash
python - <<'PY'
from pathlib import Path
import re

p = Path("src/dft_local/transport/boltzmann/ashcroft_comparison/diagnostics.py")
s = p.read_text().splitlines()

pattern = re.compile(r'f"\{[^}]+:(?:\.\d+)?[eEfFgG%][^}]*\}"')

count = 0
for i, line in enumerate(s, start=1):
    if pattern.search(line):
        count += 1
        print(f"{i}: {line.rstrip()}")

print(f"\nremaining formatted numeric cells: {count}")
PY
```

Expected at handoff:

```text
remaining formatted numeric cells: 0
```

Do not reintroduce table-cell numeric formatting like:

```python
f"{value:.8e}"
f"{value:.3f}"
```

Use typed wrappers instead.

---

## 9. Ashcroft/Vincent conductivity diagnostics

Domain:

```text
src/dft_local/transport/boltzmann/ashcroft_comparison/
```

Purpose:

```text
Compare local lattice-resolved conductivity construction against Vincent/Ashcroft reference.
Understand residual differences between local calculations and Vincent’s reported conductivity tensor.
```

Recent committed cleanup before later unit/DOS work:

```text
0737d27 Clarify Ashcroft diagnostics navigation
```

That commit added:

```text
generated left-hand hamburger contents menu
less intrusive contents label
reorganized visible narrative sections and collapsed detail sections
clearer weak/strong local calculation prose with inline Typst math
updated tests for nested sections and TOC-aware rendering
```

Targeted tests after that commit:

```text
90 passed
```

### 9.1 Page narrative structure

Desired structure:

```text
1. Local calculation check
   visible: equations and validation summary
   collapsed: detailed derivative/tensor/invariant checks

2. Velocity comparison
   visible: conclusion about Delaunay interpolation
   collapsed: raw Delaunay and k-grid evidence

3. Conductivity comparison
   visible: Fermi window, weak reconstruction, method comparison, strong/spectral check
   collapsed: raw tensor and normalisation details
```

The hamburger table of contents is generated only from diagnostic sections/subsections, not every table.

When adding diagnostics:

```text
short ProseBlock explaining the question
one compact table with method/error/conclusion
raw evidence collapsed into detail sections
use rich(...) and TypstMath(...) for formulas
wrap all semantic numeric table values as DisplayQuantity
```

### 9.2 Important functions

Core numerical functions in `core.py`:

```text
conductivity_from_velocity_grid
conductivity_from_epsilon_grid
band_indexed_strong_dc_from_velocity_grid
conductivity_830_shifted_chain_rule_from_velocity_grid
fermi_factor
fermi_window
bilinear_periodic_sample
cartesian_k_to_fractional
lattice_mode_vectors_m
vincent_delaunay_velocity_grid
```

Rendered sections in `diagnostics.py`:

```text
ashcroft_local_calculation_check
ashcroft_velocity_comparison
ashcroft_conductivity_comparison
section_conductivity_method_comparison
section_band_indexed_strong_dc
section_vincent_strong_weak_temperature_sweep
```

Tests:

```text
src/dft_local/transport/boltzmann/ashcroft_comparison/test_diagnostics.py
src/dft_local/diagnostics/test_models_business_logic.py
src/dft_local/diagnostics/test_server.py
```

### 9.3 Vincent velocity conclusion

The original `find_simplex` Delaunay interpolation gave large max sample error:

```text
find_simplex max sample error ≈ 2.43756681e+04 m/s
```

The best-adjacent simplex interpretation reduces it to roundoff:

```text
best-adjacent max sample error ≈ 1.20483578e-07 m/s
```

Reduction factor:

```text
≈ 2.023e+11
```

Conclusion:

```text
Vincent’s printed velocity samples are reproducible to roundoff by choosing adjacent Delaunay simplex gradients.
The discrepancy is a simplex-selection ambiguity at grid vertices, not a unit, k-grid, hbar, Fourier, or Gamma velocity error.
```

### 9.4 Conductivity method comparison

New comparator added before the unit handoff:

```python
conductivity_830_shifted_chain_rule_from_velocity_grid(...)
```

It compares Vincent equation 8.30-like shifted finite-field chain-rule conductivity.

Reported method comparison:

```text
Vincent target                  trace = 1.28920393e-01

weak chain-rule grid             trace = 1.34225251e-01
                                 error = +4.1148%

strong shifted Eq. 8.30          trace = 1.34245630e-01
                                 error = +4.1306%

strong spectral zero-field       trace = 1.42682063e-01
                                 error = +10.6745%
```

Interpretation:

```text
weak chain-rule ≈ shifted Eq. 8.30
both differ from Vincent by about 4.1%

strong spectral zero-field differs more strongly, about 10.7%
```

Conclusion:

```text
Vincent is not explained simply by saying “he used shifted Eq. 8.30 with finite shifts”.
The shifted Eq. 8.30 implementation agrees with weak chain-rule, not Vincent.
```

### 9.5 Strong/modal versus weak-chain

Current reported values:

```text
strong/modal trace ≈ 1.42682310e-01 S/m
weak-chain trace   ≈ 1.34225251e-01 S/m
relative gap       ≈ 6.30064649%
```

Conclusion:

```text
The modal components exactly reconstruct the strong spectral tensor.
The remaining strong/weak gap is a derivative-definition/model difference:
spectral derivative of sampled periodic occupation f0(k) versus weak chain-rule derivative f0'(E) dE/dk.
This is no longer treated as a units bug.
```

The spectral strong-zero-field result differs more because it differentiates the sampled periodic occupation:

```text
∂_{k_i}^{spectral} f0(k)
```

At 300 K, the occupation is sharp near the Fermi surface. On the finite grid, this derivative is under-resolved. Spectral differentiation is global and periodic, so it is sensitive to ringing and resolution error.

The diagnostic includes a temperature sweep showing that spectral/chain-rule mismatch decreases as temperature increases and the Fermi occupation becomes smoother.

### 9.6 Lattice-resolved/modal conductivity status

The Ashcroft comparison exposes lattice-resolved/modal conductivity information. It shows:

```text
modal components reconstruct the strong spectral tensor
dominant contributions come from first-shell modes
zero-mode response is zero because R = 0
```

Known dominant first-shell modes include:

```text
(1, -1)
(-1, 1)
```

### 9.7 Remaining 4.1% discrepancy

The residual 4.1% difference between Vincent and weak/shifted-chain-rule conductivity remains open.

Leading suspects:

```text
1. Velocity convention inside conductivity integral
2. k-space measure / endpoint convention
3. chemical potential / Fermi-level convention
4. interpolation convention for shifted quantities
5. small constants / units / degeneracy details
```

High-priority next diagnostic:

```text
Residual 4% hypothesis scan
```

Suggested compact table:

```text
method / perturbation                  trace error vs Vincent
current weak chain-rule grid            +4.11%
mu scan best value                      ...
N vs N-1 endpoint measure               ...
central velocity                        ...
spectral velocity                       ...
Delaunay velocity field                 ...
hex/BZ mask                             ...
alternative interpolation               ...
```

Priority implementations:

```text
1. Chemical potential scan
   vary mu near current value
   report best trace match and required Δmu

2. Velocity-field convention scan
   compare central finite-difference, spectral, and Vincent-style Delaunay velocity fields

3. Measure / endpoint scan
   compare N1*N2 vs (N1-1)*(N2-1) and other plausible grid weights

4. Domain / mask scan
   full rectangular reciprocal cell vs BZ hex mask vs symmetry-reduced region
```

---

## 10. Theory/notation constraints relevant to the code

The thesis/theory pipeline uses group-labelled convolution kernels and group Fourier symbols.

Right-convolution convention:

```text
(Oψ)(g) = Σ_h K_O(h) ψ(g h)
```

Fourier transform:

```text
ψ̂(Ω) = Σ_g Ω(g)† ψ(g)
```

Right-convolution symbol:

```text
Ô(Ω) = Σ_h Ω(h) K_O(h)
```

For matrix-valued symbols:

```text
Ĥ(Ω) = Σ_h Ω(h) ⊗ K_H(h)
Ŝ(Ω) = Σ_h Ω(h) ⊗ K_S(h)
```

Generalized eigenproblem:

```text
Ĥ(Ω) u_n(Ω) = E_n(Ω) Ŝ(Ω) u_n(Ω)
u_m† Ŝ(Ω) u_n = δ_mn
```

Symbol derivatives:

```text
D_i Ĥ(Ω_k) = Σ_h (D_i Ω_k(h)) ⊗ K_H(h)
D_i Ŝ(Ω_k) = Σ_h (D_i Ω_k(h)) ⊗ K_S(h)
```

Derivative hits representation matrices only, not kernel blocks.

Generalized Hellmann-Feynman:

```text
∂_i E_n = u_n† (D_i Ĥ - E_n D_i Ŝ) u_n
```

Velocity:

```text
V_i(Ω) = (1/ħ)(D_i Ĥ(Ω) - A(Ω) D_i Ŝ(Ω))
A(Ω) = Ŝ(Ω)^-1 Ĥ(Ω)
```

Physical velocity requires the group-to-physical k transform:

```text
k = T^T k_phys
∇_{k_phys} E_n = T ∇_k E_n
v_n = (1/ħ) T ∇_k E_n
```

---

## 11. Command and patching style

### 11.1 Do not use `set -euo pipefail`

The user specifically does not want it because it can kill or poison an interactive shell session.

Preferred pattern:

```bash
command
status=$?

if [ "$status" -ne 0 ]; then
  printf '\033[1;31m%s\033[0m\n' "Command failed; paste output"
else
  printf '\033[1;32m%s\033[0m\n' "Command passed"
fi
```

### 11.2 Use ANSI color sparingly

Preferred colors:

```bash
printf '\033[1;33m%s\033[0m\n' "Starting task / running check"
printf '\033[1;32m%s\033[0m\n' "Success"
printf '\033[1;31m%s\033[0m\n' "Failure; paste output"
printf '\033[1;36m%s\033[0m\n' "Useful diagnostic value"
printf '\033[1;35m%s\033[0m\n' "Section header"
```

### 11.3 Structure commands as small staged scripts

Preferred layout:

```bash
printf '\033[1;33m%s\033[0m\n' "Explain current step"

python - <<'PY'
# small targeted patch or inspection
PY

printf '\033[1;33m%s\033[0m\n' "Run focused checks"

command
status=$?

if [ "$status" -ne 0 ]; then
  printf '\033[1;31m%s\033[0m\n' "Focused checks failed; paste output"
else
  printf '\033[1;32m%s\033[0m\n' "Focused checks passed"
fi
```

### 11.4 Inspect before patch

When code shape is uncertain:

```bash
grep -nE "needle1|needle2|needle3" file.js file.py

python - <<'PY'
from pathlib import Path

p = Path("file.js")
lines = p.read_text().splitlines()

for i, line in enumerate(lines, 1):
    if "needle" in line:
        lo = max(1, i - 40)
        hi = min(len(lines), i + 100)
        for n in range(lo, hi + 1):
            print(f"{n:5d}: {lines[n-1]}")
PY
```

### 11.5 Avoid broad regex surgery

`dft-local-components.js` is large and easy to corrupt.

Safe hierarchy:

```text
best:
  replace exact known block

okay:
  find function by name and brace-match carefully

risky:
  regex over broad regions

bad:
  replace until "next def" / next class / arbitrary marker without checking
```

A bad patch once swallowed later definitions in `core.py`, temporarily breaking:

```text
ImportError: cannot import name 'LocalRegion'
```

Lesson:

```text
When patching Python functions, do not assume “next def” is safe unless the boundary is verified.
```

### 11.6 Always run syntax before full checks

For JS:

```bash
node --check src/dft_local/diagnostics/static/dft-local-components.js
node --check src/dft_local/diagnostics/static/dft-local-components.test.mjs
node --check src/dft_local/diagnostics/static/dft-local-components.browser.test.mjs
npm run js:check
```

For Python, use focused pytest before full suite when possible.

### 11.7 Browser tests

When a browser behavior changes:

```bash
npm run browser:check -- \
  --test-name-pattern "test name A|test name B|test name C"
```

Then run full:

```bash
npm run js:check
```

If browser tests hang before printing assertion bodies, run a small debug probe instead of guessing:

```text
load real page
capture pageerror / console errors
query custom element state
print status text
print whether canvas/model/control exists
```

### 11.8 Keep patches and tests in the same command block

Good command blocks usually do:

```text
1. patch source
2. patch tests
3. syntax check
4. focused check
5. full check
6. show diff/status
```

End with:

```bash
git diff -- relevant/files
git status --short
```

Do not commit automatically unless the user asks to commit or the workflow clearly says commit now.

### 11.9 Failure-guided iteration

When checks fail:

```text
read exact error
patch the smallest stale test or type issue
rerun the failed layer only
then rerun full check
```

Common failure classes:

```text
literal "\n" accidentally written into JS
TypeScript implicit any in browser-test page.evaluate callbacks
stale source-policy tests looking for old strings
server payload overriding JS fallback
broad function replacement deleting neighbouring definitions
```

---

## 12. Common checks

### 12.1 General diagnostics/server checks

```bash
python -m pytest \
  src/dft_local/diagnostics/test_server.py \
  src/dft_local/transport/bands/test_diagnostics.py \
  src/dft_local/transport/boltzmann/group_resolved/test_diagnostics.py \
  -q
```

### 12.2 Ashcroft checks

```bash
pytest -q src/dft_local/transport/boltzmann/ashcroft_comparison/test_diagnostics.py

pytest -q \
  src/dft_local/diagnostics/test_models_business_logic.py \
  src/dft_local/diagnostics/test_server.py
```

### 12.3 DisplayQuantity/render checks

```bash
pytest -q \
  src/dft_local/diagnostics/test_server.py::test_rendered_tables_have_json_copy_button \
  src/dft_local/diagnostics/test_server.py::test_rendered_table_json_copy_unwraps_display_quantities \
  src/dft_local/diagnostics/test_server.py::test_rendered_table_json_copy_unwraps_unitless_display_quantities \
  src/dft_local/diagnostics/test_server.py::test_display_quantity_uses_compact_scientific_formatting
```

### 12.4 DOS/IDOS checks

```bash
node --check src/dft_local/diagnostics/static/dft-local-components.js
node --test src/dft_local/diagnostics/static/dft-local-components.test.mjs

pytest -q \
  src/dft_local/diagnostics/test_models_business_logic.py \
  src/dft_local/transport/boltzmann/group_resolved/test_diagnostics.py
```

Audit:

```bash
python - <<'PY'
from pathlib import Path

s = Path("src/dft_local/transport/boltzmann/group_resolved/diagnostics.py").read_text()
print("dos_payload count:", s.count("dos_payload = {"))
print("np.histogram count:", s.count("np.histogram"))
PY
```

Expected:

```text
dos_payload count: 1
np.histogram count: 0
```

### 12.5 JS full checks

```bash
node --check src/dft_local/diagnostics/static/dft-local-components.js
node --test src/dft_local/diagnostics/static/dft-local-components.test.mjs
npm run js:check
```

---

## 13. Current likely next steps

### 13.1 Manual smoke after latest DOS commit

```text
group-resolved diagnostic
change nu/nv
Run
confirm DOS/IDOS curve updates from model island
confirm curve samples = 768
confirm sigma finite
confirm no stale histogram-style bars
confirm Run button patches without replacing viewer state
```

### 13.2 Viewer/K-point follow-up if needed

If Dirac crossing does not visually sit at K:

```text
add analytic-K / nearest-sample overlay
print nearest grid indices and k values
check whether sampled coordinates are reciprocal primitive phase coordinates or transformed group/irrep coordinates
```

### 13.3 Render invalidation follow-up if stale visuals appear

Add focused browser tests for:

```text
energy scale slider calls renderThreeOnce
energy zero slider calls renderThreeOnce
band toggle calls renderThreeOnce
slice slider updates overlay and calls renderThreeOnce
```

Keep no permanent render loop.

### 13.4 Ashcroft residual scan

Implement compact residual 4% hypothesis scan:

```text
mu scan
velocity convention scan
measure / endpoint scan
domain / mask scan
```

Keep it narrative and table-driven, with raw details collapsed.

### 13.5 Units cleanup

Consider moving reusable units/dimensions into `core.units`:

```text
ELECTRIC_FIELD_STRENGTH
PERCENT
common DisplayQuantity helpers
```

Do this only if values are needed outside Ashcroft diagnostics.

### 13.6 Run button status/spinner

Still useful UI trust task:

```html
<button type="submit" data-dft-run-button>Run</button>
<span data-dft-run-status aria-live="polite"></span>
```

Submit hook idea:

```html
data-on:submit="window.dftDiagnosticRunStarted?.(); @get('/d-run/...', {contentType: 'form'})"
```

SSE end:

```text
event: datastar-execute-script
data: script window.dftDiagnosticRunComplete?.()
```

JS helper idea:

```js
function setDftDiagnosticRunState(state) {
  const button = document.querySelector("[data-dft-run-button]");
  const status = document.querySelector("[data-dft-run-status]");

  if (state === "running") {
    if (button) button.disabled = true;
    if (status) status.textContent = "computing…";
    return;
  }

  if (state === "complete") {
    if (button) button.disabled = false;
    if (status) status.textContent = "updated";
    return;
  }

  if (button) button.disabled = false;
  if (status) status.textContent = "";
}

window.setDftDiagnosticRunState = setDftDiagnosticRunState;
window.dftDiagnosticRunStarted = () => setDftDiagnosticRunState("running");
window.dftDiagnosticRunComplete = () => setDftDiagnosticRunState("complete");
```

Acceptance:

```text
form has data-on:submit
form calls dftDiagnosticRunStarted
page has data-dft-run-button
page has data-dft-run-status
SSE stream includes dftDiagnosticRunComplete
JS check passes
pytest passes
```

---

## 14. Final working rules to preserve

```text
Do not use set -euo pipefail.
Inspect before patching.
Do not broad-regex dft-local-components.js.
Do not replace full interactive components on rerun.
Use updateModel(model) for model-backed custom elements.
Use DisplayQuantity for semantic numeric diagnostics.
Keep JSON copy/export raw and machine-readable.
Patch Python source-of-truth and JS fallback together for geometry.
Do geometry audit before numerical Dirac/gap search.
Do not bring back permanent Three render loop.
Run syntax before full checks.
Commit only after green checks and user approval.
```

## Typed scalar validation probe update

The finite-field validation domain has now been moved away from suffix-keyed dictionaries and onto typed scalar dataclasses using the shared unit metadata system.

Implemented typed finite-field probes:

- `FiniteFieldInputHealthProbe`
- `FiniteFieldBandCrossingHazardProbe`
- `FiniteFieldVelocityValidationProbe`
- `FiniteFieldUnitScalingProbe`
- `FiniteFieldAnalyticToyCoverageProbe`
- `FiniteFieldKConvergenceProbe`
- `FiniteFieldSymmetrySanityProbe`
- `FiniteFieldVincentReconstructionProbe`
- `FiniteFieldStrongDcValidationProbe`
- `FiniteFieldWeakDcLimitProbe`
- `FiniteFieldModeDecompositionProbe`

These probes expose scalar fields annotated with `qscalar(...)`, carry a `unit_context`, and are rendered with `diagnostic_scalar_quantity(...)` in the diagnostics table layer.

This replaces the previous pattern where physical meaning was inferred from dictionary key suffixes such as `_S_per_m`, `_m_per_s`, `_percent_error`, or `_V_per_m`.

Current finite-field validation invariant:

- `finite_field_*_probe` functions should return frozen dataclass probe objects, not plain dictionaries.
- Semantic scalar fields should be `Annotated[float, qscalar(...)]`.
- Counts, booleans, strings, shapes, and statuses can remain plain typed fields.
- Diagnostics row builders should call `diagnostic_scalar_quantity(probe, field_name)` for scalar quantities.
- `test_finite_field_validation_probes_are_not_plain_dict_returns` guards this finite-field invariant.

Current unit-handling status:

- The finite-field domain has typed scalar unit metadata and conversion-factor checks.
- The unit-scaling probe checks Hartree/eV, Bohr/Å, hbar in atomic and legacy contexts, velocity scaling, inverse-energy Fermi-window scaling, and the requirement that `mu` is converted with the Hamiltonian.
- Many finite-field toy quantities are intentionally dimensionless because they validate symbolic/synthetic models.
- Full end-to-end SI conductivity agreement across multiple `UnitContext`s is still a postponed task.

Current dataset-backed validation status:

- Finite-field input health now uses the selected dataset-backed H/S kernels when a diagnostics context is available.
- Input-health symbol checks use the degree-2 generic graphene symbol path, matching the production dataset-backed local symbol construction.
- The input-health table reports both kernel-level star defects and formed-symbol Hermiticity defects.
- Star symmetrization now shows the expected near-machine-precision Hermiticity defects in the formed H(k) and S(k) symbols.
- The remaining input-health items are visual audits rather than scalar blockers.

Current selected-band crossing hazard status:

- The band-crossing hazard section now has a production dataset-backed selected-band adjacent-gap scan.
- The scan solves the selected H/S symbol on the configured k-grid and inspects only adjacent sorted-energy gaps touching `band_index`.
- For selected band `n`, this means the adjacent gaps `(n - 1, n)` and `(n, n + 1)` when those neighbours exist.
- Crossings between unrelated band pairs are intentionally ignored because they do not directly threaten the selected band-labelled conductivity quantity.
- The hazard-point table reports only k-points whose adjacent selected-band gap falls below the configured threshold.
- The controlled periodic two-level Dirac-like toy remains only as a sanity check for the hazard-counting logic.

Latest validation check:

- Full pre-push validation passed with `358 passed, 1 xfailed`.
- Current pushed validation commits are `04fb6d2 Use dataset degree-two kernels for input health` and `e82262e Add dataset-backed band crossing hazard probe`.

Still postponed:

- Type non-finite-field validation probes, including `operator_symbol_validation_probe()` and `gd_symbol_production_validation_probe()`.
- Add a broader whole-validation-domain guard once those probes are typed.
- Revisit the downloadable table task.

