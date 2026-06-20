"""Export diagnostic results as static Typst diagnostic bundles.

This is a second renderer for the existing diagnostics data model.  Diagnostic
domains keep returning DiagnosticResult objects; this module lowers those
objects into a reproducible Typst directory containing JSON data, generated
Typst glue, and a small plotting/report library.

The generated bundle is both:
* a standalone diagnostic PDF source via diagnostics.typ
* an importable component package via components.typ
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
from typing import Any, Literal

from dft_local.core.units import DisplayQuantity
from dft_local.diagnostics.models import (
    Card,
    DiagnosticResult,
    DiagnosticSection,
    EquationBlock,
    Graph2D,
    Matrix,
    ProseBlock,
    Table,
    WebGLView,
)
from dft_local.diagnostics.render import fmt, table_json_cell_value, table_json_header_value
from dft_local.diagnostics.user_strings import RichText, TypstMath

LibMode = Literal["symlink", "vendor", "none"]


def export_typst_bundle(
    result: DiagnosticResult,
    out_dir: str | Path,
    *,
    report_id: str | None = None,
    title: str | None = None,
    provenance: dict[str, Any] | None = None,
    lib_source: str | Path | None = None,
    lib_mode: LibMode = "symlink",
) -> Path:
    """Write a Typst diagnostic bundle for ``result`` and return its path."""

    writer = _BundleWriter(
        result=result,
        out_dir=Path(out_dir),
        report_id=report_id or _slug(_plain_text(result.title)) or "diagnostics",
        title=title or _plain_text(result.title),
        provenance=provenance or default_provenance(),
        lib_source=Path(lib_source) if lib_source is not None else default_typst_lib_source(),
        lib_mode=lib_mode,
    )
    writer.write()
    return writer.out_dir


def default_provenance() -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": _git_head(),
    }


def default_typst_lib_source() -> Path:
    return Path(__file__).resolve().parents[3] / "typst-diagnostics-lib"


def _git_head() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    value = proc.stdout.strip()
    return value or None


class _BundleWriter:
    def __init__(
        self,
        *,
        result: DiagnosticResult,
        out_dir: Path,
        report_id: str,
        title: str,
        provenance: dict[str, Any],
        lib_source: Path,
        lib_mode: LibMode,
    ) -> None:
        self.result = result
        self.out_dir = out_dir
        self.report_id = report_id
        self.title = title
        self.provenance = provenance
        self.lib_source = lib_source
        self.lib_mode = lib_mode
        self.data_dir = out_dir / "data"
        self.items: list[dict[str, Any]] = []
        self.component_defs: list[str] = []
        self.report_calls: list[str] = []

    def write(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "assets").mkdir(exist_ok=True)

        self._write_lib()
        self._write_diagnostics_json()
        self._collect_result()
        self._write_manifest()
        self._write_components_typ()
        self._write_diagnostics_typ()

    def _write_lib(self) -> None:
        target = self.out_dir / "lib"
        if self.lib_mode == "none":
            target.mkdir(exist_ok=True)
            self._ensure_default_lib(target)
            return

        if self.lib_source.exists():
            if target.exists() or target.is_symlink():
                if target.is_symlink() or target.is_file():
                    target.unlink()
                else:
                    shutil.rmtree(target)
            if self.lib_mode == "symlink":
                try:
                    target.symlink_to(
                        os.path.relpath(self.lib_source, self.out_dir),
                        target_is_directory=True,
                    )
                    return
                except OSError:
                    pass
            shutil.copytree(self.lib_source, target)
            return

        target.mkdir(exist_ok=True)
        self._ensure_default_lib(target)

    def _ensure_default_lib(self, target: Path) -> None:
        bundled_source = default_typst_lib_source()
        if bundled_source.exists():
            for child in bundled_source.iterdir():
                dest = target / child.name
                if child.is_dir():
                    shutil.copytree(child, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(child, dest)
            return

        _write_fallback_typst_lib(target)

    def _write_diagnostics_json(self) -> None:
        _write_json(self.out_dir / "diagnostics.json", result_to_json_data(self.result))

    def _collect_result(self) -> None:
        for block in _result_blocks(self.result):
            self._collect_block(block, parent="root")

    def _collect_block(self, block: Any, *, parent: str) -> str:
        if isinstance(block, DiagnosticSection):
            return self._collect_section(block, parent=parent)
        if isinstance(block, ProseBlock):
            return self._collect_prose(block, parent=parent)
        if isinstance(block, EquationBlock):
            return self._collect_equation(block, parent=parent)
        if isinstance(block, Table):
            return self._collect_table(block, parent=parent)
        if isinstance(block, Graph2D):
            return self._collect_graph(block, parent=parent)
        if isinstance(block, WebGLView):
            return self._collect_webgl(block, parent=parent)
        if isinstance(block, Matrix):
            return self._collect_matrix(block, parent=parent)
        if isinstance(block, Card):
            return self._collect_card(block, parent=parent)
        return self._collect_unsupported(_block_id(block), type(block).__name__, parent=parent)

    def _collect_section(self, section: DiagnosticSection, *, parent: str) -> str:
        cid = _safe_component_name(section.id, "section")
        child_calls: list[str] = []

        if section.description:
            desc = ProseBlock(id=f"{section.id}_description", title="", markdown=section.description)
            child_calls.append(self._collect_prose(desc, parent=section.id))

        for child in _section_blocks(section):
            child_calls.append(self._collect_block(child, parent=section.id))

        level = "=" if parent == "root" else "=="
        body = "\n".join(f"  #{call}()" for call in child_calls)
        self.component_defs.append(
            f"#let {cid}() = [\n"
            f"{level} {_typst_content(section.title)}\n"
            f"{body}\n"
            f"]\n"
        )
        self.items.append(
            {
                "id": section.id,
                "kind": "section",
                "component": cid,
                "parent": parent,
                "title": _plain_text(section.title),
                "collapsed": bool(section.collapsed),
            }
        )
        if parent == "root":
            self.report_calls.append(cid)
        return cid

    def _collect_prose(self, block: ProseBlock, *, parent: str) -> str:
        cid = _safe_component_name(block.id, "prose")
        title = _typst_content(block.title)
        heading = f"*{title}*\n\n" if title else ""
        self.component_defs.append(f"#let {cid}() = [\n{heading}{_typst_content(block.markdown)}\n]\n")
        self.items.append(
            {
                "id": block.id,
                "kind": "prose",
                "component": cid,
                "parent": parent,
                "title": _plain_text(block.title),
            }
        )
        if parent == "root":
            self.report_calls.append(cid)
        return cid

    def _collect_equation(self, block: EquationBlock, *, parent: str) -> str:
        cid = _safe_component_name(block.id, "equation")
        self.component_defs.append(f"#let {cid}() = [\n{_typst_math_source(block.math)}\n]\n")
        self.items.append(
            {
                "id": block.id,
                "kind": "equation",
                "component": cid,
                "parent": parent,
                "title": getattr(block.math, "name", "") or block.id,
            }
        )
        if parent == "root":
            self.report_calls.append(cid)
        return cid

    def _collect_table(self, table: Table, *, parent: str) -> str:
        cid = _safe_component_name(table.id, "table")
        data_path = f"data/{_slug(table.id)}.json"
        _write_json(self.out_dir / data_path, table_to_json_data(table))
        self.component_defs.append(
            f"#let {cid}-data = json(\"{data_path}\")\n"
            f"#let {cid}() = diagnostic-figure(\n"
            f"  diagnostic-table({cid}-data),\n"
            f"  title: [{_typst_content(table.title)}],\n"
            f"  caption: [{_typst_content(table.description)}],\n"
            f")\n"
        )
        self._add_data_item(table.id, "table", cid, parent, data_path, table.title)
        return cid

    def _collect_graph(self, graph: Graph2D, *, parent: str) -> str:
        cid = _safe_component_name(graph.id, "graph")
        data_path = f"data/{_slug(graph.id)}.json"
        _write_json(self.out_dir / data_path, graph_to_json_data(graph))
        self.component_defs.append(
            f"#let {cid}-data = json(\"{data_path}\")\n"
            f"#let {cid}-plot() = line-graph({cid}-data)\n"
            f"#let {cid}() = diagnostic-figure(\n"
            f"  {cid}-plot(),\n"
            f"  title: [{_typst_content(graph.title)}],\n"
            f"  caption: [{_typst_content(graph.description)}],\n"
            f")\n"
        )
        self._add_data_item(graph.id, "graph2d", cid, parent, data_path, graph.title, plot_component=f"{cid}-plot")
        return cid

    def _collect_webgl(self, view: WebGLView, *, parent: str) -> str:
        cid = _safe_component_name(view.id, "view")
        data_path = f"data/{_slug(view.id)}.json"
        _write_json(self.out_dir / data_path, webgl_to_json_data(view))
        self.component_defs.append(
            f"#let {cid}-data = json(\"{data_path}\")\n"
            f"#let {cid}() = diagnostic-figure(\n"
            f"  unsupported-view({cid}-data),\n"
            f"  title: [{_typst_content(view.title)}],\n"
            f"  caption: [{_typst_content(view.description)}],\n"
            f")\n"
        )
        self._add_data_item(
            view.id,
            "webgl-placeholder",
            cid,
            parent,
            data_path,
            view.title,
            static_support="placeholder",
        )
        return cid

    def _collect_matrix(self, matrix: Matrix, *, parent: str) -> str:
        cid = _safe_component_name(matrix.id, "matrix")
        data_path = f"data/{_slug(matrix.id)}.json"
        _write_json(self.out_dir / data_path, matrix_to_json_data(matrix))
        self.component_defs.append(
            f"#let {cid}-data = json(\"{data_path}\")\n"
            f"#let {cid}() = diagnostic-figure(\n"
            f"  diagnostic-table({cid}-data),\n"
            f"  title: [{_typst_content(matrix.title)}],\n"
            f"  caption: [{_typst_content(matrix.description)}],\n"
            f")\n"
        )
        self._add_data_item(matrix.id, "matrix", cid, parent, data_path, matrix.title)
        return cid

    def _collect_card(self, card: Card, *, parent: str) -> str:
        cid = _safe_component_name(card.entity_id or card.label, "card")
        data_path = f"data/{_slug(cid)}.json"
        _write_json(
            self.out_dir / data_path,
            {
                "label": _plain_text(card.label),
                "value": _json_value(card.value),
                "status": card.status,
                "help": _plain_text(card.help),
            },
        )
        self.component_defs.append(
            f"#let {cid}-data = json(\"{data_path}\")\n"
            f"#let {cid}() = block[*{_typst_content(card.label)}:* #str({cid}-data.value)]\n"
        )
        self._add_data_item(cid, "card", cid, parent, data_path, card.label)
        return cid

    def _collect_unsupported(self, block_id: object, title: str, *, parent: str) -> str:
        cid = _safe_component_name(block_id, "unsupported")
        data_path = f"data/{_slug(cid)}.json"
        _write_json(
            self.out_dir / data_path,
            {
                "id": str(block_id),
                "title": title,
                "description": "No static Typst renderer is implemented for this diagnostic block.",
            },
        )
        self.component_defs.append(
            f"#let {cid}-data = json(\"{data_path}\")\n"
            f"#let {cid}() = unsupported-view({cid}-data)\n"
        )
        self._add_data_item(str(block_id), "unsupported", cid, parent, data_path, title)
        return cid

    def _add_data_item(
        self,
        item_id: object,
        kind: str,
        component: str,
        parent: str,
        data_path: str,
        title: object,
        **extra: Any,
    ) -> None:
        item = {
            "id": str(item_id),
            "kind": kind,
            "component": component,
            "parent": parent,
            "data": data_path,
            "title": _plain_text(title),
        }
        item.update(extra)
        self.items.append(item)
        if parent == "root":
            self.report_calls.append(component)

    def _write_manifest(self) -> None:
        _write_json(
            self.out_dir / "manifest.json",
            {
                "report_id": self.report_id,
                "title": self.title,
                "summary": _plain_text(self.result.summary),
                "provenance": self.provenance,
                "typst_lib": {
                    "mode": self.lib_mode,
                    "source": str(self.lib_source),
                },
                "items": self.items,
            },
        )

    def _write_components_typ(self) -> None:
        text = (
            "// Generated by dft_local.diagnostics.typst_bundle. Do not edit by hand.\n"
            '#import "lib/mod.typ": *\n\n'
            '#let manifest = json("manifest.json")\n\n'
            + "\n".join(self.component_defs)
            + "\n"
        )
        (self.out_dir / "components.typ").write_text(text)

    def _write_diagnostics_typ(self) -> None:
        calls = "\n\n".join(f"#{name}()" for name in self.report_calls)
        text = (
            "// Generated by dft_local.diagnostics.typst_bundle. Do not edit by hand.\n"
            '#import "components.typ": *\n\n'
            f"= {_typst_content(self.result.title)}\n\n"
            f"{_typst_content(self.result.summary)}\n\n"
            "#block[\n"
            "  *Provenance*\\\n"
            "  Created: #str(manifest.provenance.created_at)\\\n"
            "  Commit: #str(manifest.provenance.code_commit)\n"
            "]\n\n"
            f"{calls}\n"
        )
        (self.out_dir / "diagnostics.typ").write_text(text)



def _write_fallback_typst_lib(target: Path) -> None:
    """Write a minimal built-in Typst library when the project lib is absent."""

    (target / "mod.typ").write_text(
        '#import "plots.typ": *\n'
        '#import "tables.typ": *\n'
        '#import "diagnostic.typ": *\n'
    )
    (target / "diagnostic.typ").write_text(
        '#let diagnostic-figure(title: none, body, caption: none, caveats: ()) = {\n'
        '  if title != none [*#title*]\n'
        '  figure(body, caption: caption)\n'
        '  if caveats.len() > 0 {\n'
        '    block[\n'
        '      *Caveats*\n'
        '      #for caveat in caveats [- #caveat]\n'
        '    ]\n'
        '  }\n'
        '}\n\n'
        '#let diagnostic-note(body) = block[#body]\n'
    )
    (target / "tables.typ").write_text(
        '#let diagnostic-table(data) = {\n'
        '  let columns = data.headers\n'
        '  table(\n'
        '    columns: columns.len(),\n'
        '    inset: 5pt,\n'
        '    stroke: 0.5pt,\n'
        '    ..columns.map(h => [*#h*]),\n'
        '    ..data.rows.flatten().map(cell => [#str(cell)]),\n'
        '  )\n'
        '}\n'
    )
    (target / "plots.typ").write_text(
        '#import "@preview/cetz:0.3.4"\n'
        '#import "@preview/lilaq:0.4.0" as lq\n\n'
        '#let line-graph(data) = {\n'
        '  let series = data.series.map(s => (\n'
        '    label: s.name,\n'
        '    x: s.points.map(p => p.x),\n'
        '    y: s.points.map(p => p.y),\n'
        '  ))\n'
        '  lq.diagram(\n'
        '    width: 11cm,\n'
        '    height: 6cm,\n'
        '    xaxis: (label: data.x_label),\n'
        '    yaxis: (label: data.y_label),\n'
        '    ..series.map(s => lq.plot(s.x, s.y, label: s.label)),\n'
        '  )\n'
        '}\n\n'
        '#let unsupported-view(data) = block[\n'
        '  *Unsupported static view:* #data.title \\\n'
        '  #data.description\n'
        ']\n'
    )

def result_to_json_data(result: DiagnosticResult) -> dict[str, Any]:
    return {
        "kind": "diagnostic_result",
        "title": _plain_text(result.title),
        "summary": _plain_text(result.summary),
        "blocks": [_node_to_json(block) for block in _result_blocks(result)],
    }


def table_to_json_data(table: Table) -> dict[str, Any]:
    return {
        "kind": "table",
        "id": table.id,
        "title": _plain_text(table.title),
        "description": _plain_text(table.description),
        "headers": [table_json_header_value(h) for h in table.headers],
        "rows": [[table_json_cell_value(cell) for cell in row.cells] for row in table.rows],
        "numeric": sorted(int(i) for i in table.numeric),
    }


def graph_to_json_data(graph: Graph2D) -> dict[str, Any]:
    return {
        "kind": "graph2d",
        "id": graph.id,
        "title": _plain_text(graph.title),
        "description": _plain_text(graph.description),
        "x_label": _plain_text(graph.x_label),
        "y_label": _plain_text(graph.y_label),
        "interaction_channel": graph.interaction_channel,
        "series": [
            {
                "name": s.name,
                "kind": s.kind,
                "points": [
                    {
                        "x": float(p.x),
                        "y": float(p.y),
                        "entity_id": p.entity_id,
                        "label": _plain_text(p.label),
                        "meta": _json_value(dict(p.meta)),
                    }
                    for p in s.points
                ],
            }
            for s in graph.series
        ],
    }


def matrix_to_json_data(matrix: Matrix) -> dict[str, Any]:
    lookup = {(cell.i, cell.j): cell.value for cell in matrix.cells}
    return {
        "kind": "matrix",
        "id": matrix.id,
        "title": _plain_text(matrix.title),
        "description": _plain_text(matrix.description),
        "headers": [""] + [_plain_text(label) for label in matrix.col_labels],
        "rows": [
            [_plain_text(row_label)] + [_json_value(lookup.get((i, j))) for j in range(len(matrix.col_labels))]
            for i, row_label in enumerate(matrix.row_labels)
        ],
    }


def webgl_to_json_data(view: WebGLView) -> dict[str, Any]:
    return {
        "kind": "webgl",
        "id": view.id,
        "title": _plain_text(view.title),
        "description": _plain_text(view.description),
        "payload": _json_value(view.payload),
        "static_support": "placeholder",
    }


def _result_blocks(result: DiagnosticResult) -> tuple[Any, ...]:
    return (
        tuple(getattr(result, "body", ()) or ())
        + tuple(getattr(result, "markdowns", ()) or ())
        + tuple(getattr(result, "cards", ()) or ())
        + tuple(getattr(result, "sections", ()) or ())
        + tuple(getattr(result, "matrices", ()) or ())
        + tuple(getattr(result, "webgl", ()) or ())
        + tuple(getattr(result, "graphs", ()) or ())
        + tuple(getattr(result, "tables", ()) or ())
        + tuple(ProseBlock(id=f"note_{i}", title="Note", markdown=note) for i, note in enumerate(getattr(result, "notes", ()) or ()))
    )


def _section_blocks(section: DiagnosticSection) -> tuple[Any, ...]:
    if getattr(section, "body", ()):
        return tuple(section.body)
    return (
        tuple(getattr(section, "markdowns", ()) or ())
        + tuple(getattr(section, "math_blocks", ()) or ())
        + tuple(getattr(section, "cards", ()) or ())
        + tuple(getattr(section, "tables", ()) or ())
        + tuple(getattr(section, "sections", ()) or ())
    )


def _node_to_json(node: Any) -> dict[str, Any]:
    if isinstance(node, DiagnosticSection):
        return {
            "kind": "section",
            "id": node.id,
            "title": _plain_text(node.title),
            "description": _plain_text(node.description),
            "collapsed": bool(node.collapsed),
            "children": [_node_to_json(child) for child in _section_blocks(node)],
        }
    if isinstance(node, ProseBlock):
        return {"kind": "prose", "id": node.id, "title": _plain_text(node.title), "markdown": _plain_text(node.markdown)}
    if isinstance(node, EquationBlock):
        return {"kind": "equation", "id": node.id, "math": _typst_math_source(node.math)}
    if isinstance(node, Table):
        return table_to_json_data(node)
    if isinstance(node, Graph2D):
        return graph_to_json_data(node)
    if isinstance(node, WebGLView):
        return webgl_to_json_data(node)
    if isinstance(node, Matrix):
        return matrix_to_json_data(node)
    if isinstance(node, Card):
        return {
            "kind": "card",
            "label": _plain_text(node.label),
            "value": _json_value(node.value),
            "status": node.status,
            "help": _plain_text(node.help),
        }
    return {"kind": type(node).__name__, "value": _json_value(node)}


def _plain_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, RichText):
        return "".join(_plain_text(part) for part in value.parts)
    if isinstance(value, TypstMath):
        return value.source
    return str(value)


def _typst_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, RichText):
        return "".join(_typst_content(part) for part in value.parts)
    if isinstance(value, TypstMath):
        return _typst_math_source(value)
    return _escape_typst_text(str(value))


def _typst_math_source(value: TypstMath) -> str:
    return value.source


def _escape_typst_text(value: str) -> str:
    replacements = {
        "\\": "\\\\",
        "#": "\\#",
        "[": "\\[",
        "]": "\\]",
    }
    out = value
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _json_value(value: Any) -> Any:
    if isinstance(value, DisplayQuantity):
        return float(value.value)
    if isinstance(value, (RichText, TypstMath)):
        return _plain_text(value)
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return fmt(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n")


def _slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "item"


def _safe_component_name(value: object, fallback: str) -> str:
    slug = _slug(value).replace("-", "_").replace(".", "_")
    if not slug:
        slug = fallback
    if slug[0].isdigit():
        slug = f"{fallback}_{slug}"
    return slug


def _block_id(block: Any) -> object:
    return getattr(block, "id", type(block).__name__)
