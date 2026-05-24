from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from . import (
    SparseDataset,
    NearestNeighbourGraph,
    EdgeDirections,
    EdgeGroupLabels,
    GdKernelArrays,
    SymbolPair,
    DenseMatrixDiagnostics,
    hermitian_part,
)


@dataclass(frozen=True)
class DiagnosticsState:
    data: SparseDataset
    geom: NearestNeighbourGraph
    edges: EdgeDirections
    labels: EdgeGroupLabels

    KH: GdKernelArrays
    KS: GdKernelArrays

    KH_avg: GdKernelArrays
    KS_avg: GdKernelArrays

    KH_avg_star: GdKernelArrays
    KS_avg_star: GdKernelArrays


def build_state(root: str | Path) -> DiagnosticsState:
    data = SparseDataset.load(Path(root))

    geom = NearestNeighbourGraph.from_positions(data.metadata.positions)
    edges = EdgeDirections.from_geometry(geom)
    labels = EdgeGroupLabels.from_geometry(geom, edges)

    KH = GdKernelArrays.from_anchored(data.H, labels, matrix_name="H anchored")
    KS = GdKernelArrays.from_anchored(data.S, labels, matrix_name="S anchored")

    KH_avg = GdKernelArrays.from_average(data.H, labels, matrix_name="H average")
    KS_avg = GdKernelArrays.from_average(data.S, labels, matrix_name="S average")

    KH_avg_star = KH_avg.star_symmetrised(matrix_name="H average star")
    KS_avg_star = KS_avg.star_symmetrised(matrix_name="S average star")

    return DiagnosticsState(
        data=data,
        geom=geom,
        edges=edges,
        labels=labels,
        KH=KH,
        KS=KS,
        KH_avg=KH_avg,
        KS_avg=KS_avg,
        KH_avg_star=KH_avg_star,
        KS_avg_star=KS_avg_star,
    )

CSS = """
:root {
  --bg: #fafafa;
  --fg: #161616;
  --muted: #666;
  --line: #d9d9d9;
  --card: #ffffff;
  --accent: #254f7a;
  --bad: #8a1f1f;
  --good: #236b3a;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
}

header {
  border-bottom: 1px solid var(--line);
  background: var(--card);
  padding: 1rem 1.5rem;
}

main {
  max-width: 1100px;
  margin: 0 auto;
  padding: 1.25rem;
}

nav a {
  display: inline-block;
  margin-right: 1rem;
  color: var(--accent);
  text-decoration: none;
  font-weight: 600;
}

nav a:hover {
  text-decoration: underline;
}

h1, h2, h3 {
  line-height: 1.2;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 1rem;
}

.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 1rem;
}

.metric {
  font-size: 1.55rem;
  font-weight: 700;
}

.label {
  color: var(--muted);
  font-size: 0.9rem;
}

table {
  border-collapse: collapse;
  width: 100%;
  background: var(--card);
  margin: 1rem 0;
}

th {
  text-align: left;
  border-bottom: 2px solid var(--fg);
  padding: 0.45rem 0.55rem;
}

td {
  border-bottom: 1px solid var(--line);
  padding: 0.45rem 0.55rem;
  vertical-align: top;
}

td.num, th.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

code {
  background: #eee;
  padding: 0.1rem 0.25rem;
  border-radius: 4px;
}

.good { color: var(--good); font-weight: 700; }
.bad { color: var(--bad); font-weight: 700; }

.small {
  color: var(--muted);
  font-size: 0.9rem;
}

form.inline {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: end;
  margin: 1rem 0;
}

input {
  padding: 0.35rem 0.45rem;
  border: 1px solid var(--line);
  border-radius: 5px;
}

button {
  padding: 0.4rem 0.75rem;
  border: 1px solid var(--accent);
  background: var(--accent);
  color: white;
  border-radius: 5px;
}
"""

VIEWER_CSS = """
html, body {
  margin: 0;
  height: 100%;
  overflow: hidden;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

#app {
  display: grid;
  grid-template-columns: 1fr 320px;
  height: 100vh;
}

#viewer {
  width: 100%;
  height: 100%;
  background: #101216;
}

#panel {
  border-left: 1px solid #ddd;
  padding: 1rem;
  background: #fafafa;
  overflow: auto;
}

h1 {
  font-size: 1.1rem;
  margin: 0 0 0.75rem 0;
}

dl {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.35rem 0.75rem;
}

dt {
  color: #666;
}

dd {
  margin: 0;
  font-variant-numeric: tabular-nums;
}

a {
  color: #254f7a;
  font-weight: 600;
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

.small {
  color: #666;
  font-size: 0.9rem;
}
"""


def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if isinstance(x, (np.floating, float)):
        x = float(x)
        if not np.isfinite(x):
            return "—"
        if x == 0:
            return "0"
        ax = abs(x)
        if ax < 1e-3 or ax >= 1e4:
            return f"{x:.{digits}e}"
        return f"{x:.{digits}g}"
    return escape(str(x))


def page(title: str, body: str) -> HTMLResponse:
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>{CSS}</style>
</head>
<body>
<header>
  <h1>{escape(title)}</h1>
  <nav>
    <a href="/">Overview</a>
    <a href="/geometry">Geometry</a>
    <a href="/kernels">Kernels</a>
    <a href="/symbol?k1=0.1&k2=0.2">Symbol</a>
    <a href="/viewer">Viewer</a>
  </nav>
</header>
<main>
{body}
</main>
</body>
</html>"""
    return HTMLResponse(html)


def metric_card(label: str, value: Any, note: str = "") -> str:
    return f"""
<div class="card">
  <div class="label">{escape(label)}</div>
  <div class="metric">{fmt(value)}</div>
  <div class="small">{escape(note)}</div>
</div>
"""


def dict_table(d: dict[str, Any]) -> str:
    rows = []
    for key, value in d.items():
        rows.append(
            f"<tr><th>{escape(str(key))}</th><td class='num'>{fmt(value)}</td></tr>"
        )
    return "<table>" + "\n".join(rows) + "</table>"


def rows_table(headers: list[str], rows: list[list[Any]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    hs = "".join(
        f"<th class='{'num' if i in numeric else ''}'>{escape(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        tds = "".join(
            f"<td class='{'num' if i in numeric else ''}'>{fmt(v)}</td>"
            for i, v in enumerate(row)
        )
        body.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{hs}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def kernel_summary_row(name: str, K: GdKernelArrays) -> list[Any]:
    d = K.diagnostics()
    s = K.star_defect()

    return [
        name,
        d["support_size"],
        d["num_even"],
        d["num_odd"],
        s["num_missing_inverse"],
        s["star_defect_mean"],
        s["star_defect_max"],
    ]


def symbol_diag_row(name: str, K: GdKernelArrays, k1: float, k2: float) -> list[Any]:
    A = K.symbol_generic(k1, k2)
    d = DenseMatrixDiagnostics.from_dense_matrix(A, name=name)

    return [
        name,
        d.shape,
        d.norm,
        d.hermitian_defect_rel,
    ]


def local_symbol_rows(state: DiagnosticsState, k1: float, k2: float) -> list[list[Any]]:
    return [
        symbol_diag_row("H anchored", state.KH, k1, k2),
        symbol_diag_row("H average", state.KH_avg, k1, k2),
        symbol_diag_row("H average star", state.KH_avg_star, k1, k2),
        symbol_diag_row("S anchored", state.KS, k1, k2),
        symbol_diag_row("S average", state.KS_avg, k1, k2),
        symbol_diag_row("S average star", state.KS_avg_star, k1, k2),
    ]

def viewer_payload(state: DiagnosticsState) -> dict:
    geom = state.geom
    labels = state.labels

    atoms = []
    for a in range(geom.natoms):
        atoms.append({
            "id": int(a),
            "symbol": str(state.data.metadata.symbols[a]),
            "x": float(geom.positions[a, 0]),
            "y": float(geom.positions[a, 1]),
            "z": float(geom.positions[a, 2]),
            "m": int(labels.m[a]),
            "n": int(labels.n[a]),
            "eps": int(labels.eps[a]),
            "degree": int(geom.degree[a]),
        })

    bonds = []
    seen = set()
    for a in range(geom.natoms):
        for b in geom.neighbours(a):
            b = int(b)
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            bonds.append({"a": int(a), "b": int(b)})

    return {
        "natoms": int(geom.natoms),
        "atoms": atoms,
        "bonds": bonds,
        "anchor_atom": int(labels.anchor_atom),
    }

def create_app(root: str | Path = "test_run/run_dir/data") -> FastAPI:
    app = FastAPI(title="DFT local diagnostics")
    state = build_state(root)
    app.state.diagnostics_state = state

    @app.get("/", response_class=HTMLResponse)
    def overview():
        s: DiagnosticsState = app.state.diagnostics_state
        gdiag = s.geom.diagnostics()
        ldiag = s.labels.diagnostics()

        body = f"""
<section class="grid">
  {metric_card("Atoms", s.data.metadata.natoms)}
  {metric_card("Basis size", s.data.metadata.nbasis)}
  {metric_card("Channels / atom", s.data.basis.nchannels)}
  {metric_card("Anchor atom", s.labels.anchor_atom)}
  {metric_card("Bulk atoms", gdiag["num_bulk_atoms"])}
  {metric_card("Core bulk atoms", gdiag["num_core_bulk_atoms"])}
</section>

<h2>Label diagnostics</h2>
{dict_table({
    "visited": ldiag["visited_count"],
    "unvisited": ldiag["unvisited_count"],
    "m_min": ldiag["m_min"],
    "m_max": ldiag["m_max"],
    "n_min": ldiag["n_min"],
    "n_max": ldiag["n_max"],
})}

<h2>Kernel summary</h2>
{rows_table(
    ["kernel", "support", "even", "odd", "missing inverses", "mean star defect", "max star defect"],
    [
        kernel_summary_row("H anchored", s.KH),
        kernel_summary_row("H average", s.KH_avg),
        kernel_summary_row("H average star", s.KH_avg_star),
    ],
    numeric={1,2,3,4,5,6},
)}
"""
        return page("DFT local diagnostics", body)

    @app.get("/geometry", response_class=HTMLResponse)
    def geometry():
        s: DiagnosticsState = app.state.diagnostics_state
        gdiag = s.geom.diagnostics()
        ediag = s.edges.diagnostics(s.geom)
        ldiag = s.labels.diagnostics()

        degree_rows = [
            [degree, count]
            for degree, count in sorted(gdiag["degree_counts"].items())
        ]

        pos_err = ldiag.get("position_reconstruction_errors") or ldiag.get("positino_reconstruction_errors", {})

        body = f"""
<h2>Nearest-neighbour graph</h2>
<section class="grid">
  {metric_card("Bond length a0", gdiag["a0"], "Estimated median nearest-neighbour distance")}
  {metric_card("Cutoff", gdiag["cutoff"])}
  {metric_card("Bulk atoms", gdiag["num_bulk_atoms"])}
  {metric_card("Core bulk atoms", gdiag["num_core_bulk_atoms"])}
</section>

<h3>Degree counts</h3>
{rows_table(["degree", "count"], degree_rows, numeric={0,1})}

<h2>Edge directions</h2>
{dict_table({
    "anchor_atom": ediag["anchor_atom"],
    "alignment_min": ediag["alignment_min"],
    "alignment_median": ediag["alignment_median"],
    "alignment_max": ediag["alignment_max"],
})}

<h2>Group labels</h2>
{dict_table({
    "visited_count": ldiag["visited_count"],
    "unvisited_count": ldiag["unvisited_count"],
    "m_min": ldiag["m_min"],
    "m_max": ldiag["m_max"],
    "n_min": ldiag["n_min"],
    "n_max": ldiag["n_max"],
    "position_error_max": pos_err.get("max"),
    "position_error_mean": pos_err.get("mean"),
})}
"""
        return page("Geometry diagnostics", body)

    @app.get("/kernels", response_class=HTMLResponse)
    def kernels():
        s: DiagnosticsState = app.state.diagnostics_state

        rows = [
            kernel_summary_row("H anchored", s.KH),
            kernel_summary_row("H average", s.KH_avg),
            kernel_summary_row("H average star", s.KH_avg_star),
            kernel_summary_row("S anchored", s.KS),
            kernel_summary_row("S average", s.KS_avg),
            kernel_summary_row("S average star", s.KS_avg_star),
        ]

        worst = s.KH_avg.star_defect_table_filtered(min_norm=1e-2, max_radius=3).head(20)
        worst_rows = [
            [
                int(r.m), int(r.n), int(r.eps),
                r.norm, r.inv_norm, r.star_error, r.star_relative_error,
            ]
            for r in worst.itertuples()
        ]

        body = f"""
<h2>Kernel star defects</h2>
{rows_table(
    ["kernel", "support", "even", "odd", "missing inverses", "mean star defect", "max star defect"],
    rows,
    numeric={1,2,3,4,5,6},
)}

<h2>Worst local star defects for averaged H</h2>
<p class="small">Filtered by <code>min_norm=1e-2</code> and <code>max_radius=3</code>.</p>
{rows_table(
    ["m", "n", "eps", "norm", "inv norm", "star error", "relative"],
    worst_rows,
    numeric={0,1,2,3,4,5,6},
)}
"""
        return page("Kernel diagnostics", body)

    @app.get("/symbol", response_class=HTMLResponse)
    def symbol(
        k1: float = Query(0.1),
        k2: float = Query(0.2),
    ):
        s: DiagnosticsState = app.state.diagnostics_state

        pair = SymbolPair(s.KH_avg_star, s.KS_avg_star, k1, k2, name="average star")
        local = pair.form()
        E = local.energies()

        Hdiag = DenseMatrixDiagnostics.from_dense_matrix(local.Hk, name="H(k)")
        Sdiag = DenseMatrixDiagnostics.from_dense_matrix(local.Sk, name="S(k)", check_eigenvalues=True)

        energy_rows = [[i, e] for i, e in enumerate(E)]

        body = f"""
<form class="inline" method="get" action="/symbol">
  <label>k1<br><input name="k1" value="{fmt(k1)}"></label>
  <label>k2<br><input name="k2" value="{fmt(k2)}"></label>
  <button type="submit">Update</button>
</form>

<h2>Symbol diagnostics</h2>
{rows_table(
    ["symbol", "shape", "norm", "Hermitian defect"],
    local_symbol_rows(s, k1, k2),
    numeric={2,3},
)}

<h2>Effective problem used for energies</h2>
{dict_table({
    "H hermitian defect": Hdiag.hermitian_defect_rel,
    "S hermitian defect": Sdiag.hermitian_defect_rel,
    "S eig min": Sdiag.eig_min,
    "S eig max": Sdiag.eig_max,
    "S condition": Sdiag.condition_number_abs,
})}

<h2>Energies</h2>
{rows_table(["band", "energy / eV"], energy_rows, numeric={0,1})}
"""
        return page("Symbol diagnostics", body)


    @app.get("/viewer/data")
    def viewer_data():
        s: DiagnosticsState = app.state.diagnostics_state
        return JSONResponse(viewer_payload(s))


    @app.get("/viewer", response_class=HTMLResponse)
    def viewer():
        html = """<!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Graphene viewer</title>
      <style>""" + f'{VIEWER_CSS}' + """</style>

      <script type="importmap">
      {
        "imports": {
          "three": "https://cdn.jsdelivr.net/npm/three@0.184.0/build/three.module.js",
          "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.184.0/examples/jsm/"
        }
      }
      </script>
    </head>
    <body>
      <div id="app">
        <div id="viewer"></div>
        <aside id="panel">
          <h1>Graphene viewer</h1>
          <p class="small">Drag to rotate. Scroll to zoom. Click an atom.</p>
          <div id="atom-info">No atom selected.</div>
          <p><a href="/">Back to diagnostics</a></p>
        </aside>
      </div>

      <script type="module">
        import * as THREE from "three";
        import { OrbitControls } from "three/addons/controls/OrbitControls.js";

        const container = document.getElementById("viewer");
        const info = document.getElementById("atom-info");

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x101216);

        const camera = new THREE.PerspectiveCamera(
          50,
          container.clientWidth / container.clientHeight,
          0.1,
          10000
        );

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        const light = new THREE.DirectionalLight(0xffffff, 1.0);
        light.position.set(0, 0, 100);
        scene.add(light);

        const ambient = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambient);

        const data = await fetch("/viewer/data").then(r => r.json());

        const positions = data.atoms.map(a => new THREE.Vector3(a.x, a.y, a.z));
        const center = new THREE.Vector3();
        for (const p of positions) center.add(p);
        center.multiplyScalar(1 / positions.length);

        for (const p of positions) p.sub(center);

        const atomGroup = new THREE.Group();
        const bondGroup = new THREE.Group();
        scene.add(bondGroup);
        scene.add(atomGroup);

        const atomMaterialA = new THREE.MeshStandardMaterial({ color: 0x5dade2, roughness: 0.45 });
        const atomMaterialB = new THREE.MeshStandardMaterial({ color: 0xf5b041, roughness: 0.45 });
        const anchorMaterial = new THREE.MeshStandardMaterial({ color: 0xe74c3c, roughness: 0.35 });
        const selectedMaterial = new THREE.MeshStandardMaterial({ color: 0x2ecc71, roughness: 0.25 });
        const bondMaterial = new THREE.LineBasicMaterial({ color: 0x888888 });

        const sphereGeometry = new THREE.SphereGeometry(0.22, 16, 16);
        const atomMeshes = new Map();
        let selected = null;
        let selectedOldMaterial = null;

        for (let i = 0; i < data.atoms.length; i++) {
          const atom = data.atoms[i];
          const material =
            atom.id === data.anchor_atom ? anchorMaterial :
            atom.eps === 0 ? atomMaterialA : atomMaterialB;

          const mesh = new THREE.Mesh(sphereGeometry, material);
          mesh.position.copy(positions[i]);
          mesh.userData.atom = atom;
          atomGroup.add(mesh);
          atomMeshes.set(atom.id, mesh);
        }

        const bondPositions = [];
        for (const bond of data.bonds) {
          const pa = positions[bond.a];
          const pb = positions[bond.b];
          bondPositions.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z);
        }

        const bondGeometry = new THREE.BufferGeometry();
        bondGeometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(bondPositions, 3)
        );
        const bondLines = new THREE.LineSegments(bondGeometry, bondMaterial);
        bondGroup.add(bondLines);

        camera.position.set(0, 0, 85);
        controls.target.set(0, 0, 0);
        controls.update();

        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        let hovered = null;
        let hoveredOldMaterial = null;

        const hoveredMaterial = new THREE.MeshStandardMaterial({
          color: 0xffffff,
          roughness: 0.2
        });

        function atomHtml(atom, title = "Atom") {
          return `
            <h1>${title} ${atom.id}</h1>
            <dl>
              <dt>symbol</dt><dd>${atom.symbol}</dd>
              <dt>degree</dt><dd>${atom.degree}</dd>
              <dt>eps</dt><dd>${atom.eps}</dd>
              <dt>m</dt><dd>${atom.m}</dd>
              <dt>n</dt><dd>${atom.n}</dd>
              <dt>x</dt><dd>${atom.x.toFixed(6)}</dd>
              <dt>y</dt><dd>${atom.y.toFixed(6)}</dd>
              <dt>z</dt><dd>${atom.z.toFixed(6)}</dd>
            </dl>
            <p><a href="/atom/${atom.id}">Open atom diagnostics</a></p>
          `;
        }
        function setPanelForAtom(atom, mode) {
          const title = mode === "hover" ? "Hovered atom" : "Selected atom";
          info.innerHTML = atomHtml(atom, title);
        }

        function clearHoverPanel() {
          if (selected) {
            setPanelForAtom(selected.userData.atom, "selected");
          } else {
            info.innerHTML = "No atom selected.";
          }
        }

        function setHovered(mesh) {
          if (hovered === mesh) {
            return;
          }

          if (hovered && hovered !== selected) {
            hovered.material = baseMaterialForAtom(hovered.userData.atom);
          }

          hovered = mesh;

          if (hovered) {
            if (hovered !== selected) {
              hovered.material = hoveredMaterial;
            }
            setPanelForAtom(hovered.userData.atom, "hover");
            renderer.domElement.style.cursor = "pointer";
          } else {
            clearHoverPanel();
            renderer.domElement.style.cursor = "default";
          }
        }


        function pickAtom(event) {
          const rect = renderer.domElement.getBoundingClientRect();

          mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
          mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

          raycaster.setFromCamera(mouse, camera);
          const hits = raycaster.intersectObjects(atomGroup.children, false);

          if (hits.length > 0) {
            return hits[0].object;
          }

          return null;
        }

        function baseMaterialForAtom(atom) {
          if (atom.id === data.anchor_atom) {
            return anchorMaterial;
          }
          return atom.eps === 0 ? atomMaterialA : atomMaterialB;
        }

        function selectAtom(mesh) {
          if (selected && selected !== hovered) {
            selected.material = baseMaterialForAtom(selected.userData.atom);
          }

          selected = mesh;
          selectedOldMaterial = baseMaterialForAtom(mesh.userData.atom);

          mesh.material = selectedMaterial;

          const atom = mesh.userData.atom;
          setPanelForAtom(atom, "selected");
        }
        

        renderer.domElement.addEventListener("click", event => {
          const rect = renderer.domElement.getBoundingClientRect();

          mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
          mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

          raycaster.setFromCamera(mouse, camera);
          const hits = raycaster.intersectObjects(atomGroup.children, false);

          if (hits.length > 0) {
            selectAtom(hits[0].object);
          }
        });

        window.addEventListener("resize", () => {
          camera.aspect = container.clientWidth / container.clientHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(container.clientWidth, container.clientHeight);
        });

        renderer.domElement.addEventListener("mousemove", event => {
          const mesh = pickAtom(event);
          setHovered(mesh);
        });

        renderer.domElement.addEventListener("click", event => {
          const mesh = pickAtom(event);

          if (mesh) {
            selectAtom(mesh);
          }
        });

        function animate() {
          requestAnimationFrame(animate);
          controls.update();
          renderer.render(scene, camera);
        }

        animate();
      </script>
    </body>
    </html>"""
        return HTMLResponse(html)

    @app.get("/atom/{atom_id}", response_class=HTMLResponse)
    def atom_detail(atom_id: int):
        s: DiagnosticsState = app.state.diagnostics_state

        if atom_id < 0 or atom_id >= s.geom.natoms:
            return page("Atom not found", f"<p>No atom {atom_id}</p>")

        g = s.labels.element(atom_id)
        ns = s.geom.neighbours(atom_id)

        neighbour_rows = []
        for b in ns:
            b = int(b)
            h = s.labels.relative(atom_id, b)
            neighbour_rows.append([
                b,
                int(s.labels.eps[b]),
                h.m,
                h.n,
                h.eps,
                float(np.linalg.norm(s.geom.positions[b] - s.geom.positions[atom_id])),
            ])

        body = f"""
    <h2>Atom {atom_id}</h2>
    <section class="grid">
      {metric_card("symbol", s.data.metadata.symbols[atom_id])}
      {metric_card("degree", s.geom.degree[atom_id])}
      {metric_card("m", g.m)}
      {metric_card("n", g.n)}
      {metric_card("eps", g.eps)}
    </section>

    <h2>Neighbours</h2>
    {rows_table(
        ["atom", "eps", "rel m", "rel n", "rel eps", "distance"],
        neighbour_rows,
        numeric={0,1,2,3,4,5},
    )}

    <p><a href="/viewer">Back to viewer</a></p>
    """
        return page(f"Atom {atom_id}", body)

    return app


app = create_app()
