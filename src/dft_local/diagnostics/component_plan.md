# Diagnostic component plan

The diagnostics app should be an interactive scientific exploration surface.

## State model

Separate three kinds of state:

1. Diagnostic data state
   - owned by the server
   - delivered as JSON payloads in the page
   - later updated by Datastar/SSE

2. Shared interaction state
   - selected entity
   - hovered entity
   - active interaction channel
   - highlighted series/entity

3. Component-local view state
   - zoom
   - pan
   - visible viewport
   - drag state
   - cached scales
   - local reset button state

Datastar signals should own shared interaction state and server-patched data state.
Web components should own high-frequency pointer handling and local view state.

## Component policy

Use reusable Web Components.

Initial components:

- `dft-line-graph`
- `dft-kspace-plot`
- `dft-data-table`
- `dft-entity-detail`

Components read JSON payloads from script tags with `type="application/json"`.

Example shape:

    <script type="application/json" id="data-band_path">
      {...}
    </script>
    <dft-line-graph data-source="data-band_path"></dft-line-graph>

## Interaction policy

Diagnostics already carry entity ids:

- `GraphPoint.entity_id`
- `TableRow.entity_id`
- `Card.entity_id`
- `interaction_channel`

These should become the public interaction layer.

Shared signal names:

- `hoveredEntity`
- `selectedEntity`
- `activeChannel`
- `highlightedSeries`
- `dataVersion`

A table row and a graph line should communicate through entity ids, not through
direct references to each other.

## Datastar policy

Use Datastar for:

- reactive shared signals
- server-patched data state
- later SSE updates

Do not use Datastar for every mousemove or wheel event.

Use component-local event handling for:

- zoom
- pan
- hit testing
- drag state
- redraw scheduling

## Data update policy

Initial page render:

- server embeds JSON data
- component reads JSON
- component initialises local view state

Later SSE update:

- server patches a Datastar data signal
- component receives new data
- component preserves compatible view state
- component exposes a reset-view button

## K-space plotting policy

K-space plots must use Cartesian display coordinates.

A K-space plot should show reference geometry:

- selected path
- Gamma, K, M points
- primitive-cell boundary
- Brillouin-zone hexagon

K-space plot view state should default to equal aspect ratio.

## Testing policy

Introducing JavaScript requires JavaScript tests.

Minimum checks:

- `node --check` for syntax
- `tsc --noEmit --allowJs --checkJs` for JSDoc type checking
- `node --test` for pure helper functions
- Python render tests for component mounting and payload wiring

No JavaScript build step should be required initially. Use plain `.js` with
JSDoc types and `// @ts-check`.

## First implementation milestone

1. Add static JS/CSS serving.
2. Add `dft-local-components.js`.
3. Add `dft-local.css`.
4. Render graph payloads as JSON script tags.
5. Mount `dft-line-graph` and `dft-kspace-plot`.
6. Add zoom, pan, reset view.
7. Add hover/selection signals.
8. Add JS syntax/type checks to the pre-push hook.
