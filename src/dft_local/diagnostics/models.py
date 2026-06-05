"""Data models for the diagnostics panel.

The diagnostics panel treats every screen as a small scientific question.  The
question is described by a :class:`DiagnosticSpec`; running it returns a
:class:`DiagnosticResult`.  Results are deliberately plain data, not HTML, so
that diagnostics are easy to test and easy to render in multiple ways.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Literal, Mapping, TypeAlias
import json

from dft_local.diagnostics.user_strings import UserString

InputKind = Literal["int", "float", "str", "select", "bool"]
Status = Literal["ok", "warn", "bad", "neutral"]
GraphKind = Literal["line", "points", "line_points"]


class InputParseError(ValueError):
    """Raised when query-string or clipboard inputs cannot be parsed."""


@dataclass(frozen=True, slots=True)
class InputSpec:
    """Describe one user-editable diagnostic input.

    Parameters
    ----------
    name:
        Machine-readable input name.  This becomes the query-string key.
    label:
        Human-readable label shown in the control panel.
    kind:
        Type used for parsing and rendering.
    default:
        Default value used when no query-string value is supplied.
    help:
        Short explanation shown below the control.
    min_value, max_value:
        Optional numeric bounds.  These are applied during parsing for ``int``
        and ``float`` inputs.
    options:
        For ``select`` inputs, a tuple of ``(value, label)`` pairs.
    """

    name: str
    label: UserString
    kind: InputKind
    default: Any
    help: UserString = ""
    min_value: float | None = None
    max_value: float | None = None
    options: tuple[tuple[str, str], ...] = ()

    def parse(self, raw: Any | None) -> Any:
        """Parse one raw value from query parameters or clipboard JSON."""
        if raw is None or raw == "":
            value = self.default
        else:
            try:
                match self.kind:
                    case "int":
                        value = int(raw)
                    case "float":
                        value = float(raw)
                    case "bool":
                        if isinstance(raw, bool):
                            value = raw
                        else:
                            value = str(raw).lower() in {"1", "true", "yes", "on"}
                    case "select":
                        value = str(raw)
                    case "str":
                        value = str(raw)
                    case _:
                        raise InputParseError(f"Unsupported input kind: {self.kind}")
            except Exception as exc:  # noqa: BLE001 - convert to useful user error
                raise InputParseError(f"Could not parse {self.name}={raw!r}") from exc

        if self.kind in {"int", "float"}:
            x = float(value)
            if self.min_value is not None and x < self.min_value:
                raise InputParseError(f"{self.name}={value!r} below minimum {self.min_value}")
            if self.max_value is not None and x > self.max_value:
                raise InputParseError(f"{self.name}={value!r} above maximum {self.max_value}")

        if self.kind == "select" and self.options:
            allowed = {v for v, _label in self.options}
            if value not in allowed:
                raise InputParseError(f"{self.name}={value!r} not in {sorted(allowed)}")

        return value

    def serialize(self, value: Any) -> Any:
        """Return a JSON-compatible representation of an input value."""
        if self.kind == "bool":
            return bool(value)
        if self.kind == "int":
            return int(value)
        if self.kind == "float":
            return float(value)
        return str(value)


@dataclass(frozen=True, slots=True)
class Card:
    """Small status card shown near the top of a diagnostic."""

    label: UserString
    value: Any
    status: Status = "neutral"
    help: UserString = ""
    entity_id: str | None = None
    interaction_channel: str | None = None


@dataclass(frozen=True, slots=True)
class MarkdownBlock:
    id: str
    title: UserString
    markdown: UserString


@dataclass(frozen=True, slots=True)
class TableRow:
    """One table row, optionally linked to a selectable diagnostic entity."""

    cells: tuple[Any, ...]
    entity_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProseBlock:
    id: str
    title: UserString
    markdown: UserString


@dataclass(frozen=True, slots=True)
class EquationBlock:
    """A display-style equation used as document content.

    This is not a section, card, or foldable diagnostic unit. It is just a
    centered block equation in the surrounding prose.
    """

    id: str
    math: TypstMath


# Backwards-compatible names while diagnostics migrate to the document model.
MarkdownBlock = ProseBlock
TypstMathBlock = EquationBlock
ParagraphBlock = ProseBlock
DocumentBlock: TypeAlias = Any


@dataclass(frozen=True, slots=True)
class Table:
    """Generic sortable table output."""

    id: str
    title: UserString
    description: UserString
    headers: tuple[UserString, ...]
    rows: tuple[TableRow, ...]
    numeric: frozenset[int] = frozenset()
    interaction_channel: str | None = None


@dataclass(frozen=True, slots=True)
class GraphPoint:
    """One graph point, optionally tied to a table row or detail record."""

    x: float
    y: float
    entity_id: str | None = None
    label: UserString = ""
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphSeries:
    """One line/point series for the reusable canvas graph component."""

    name: str
    points: tuple[GraphPoint, ...]
    kind: GraphKind = "line"


@dataclass(frozen=True, slots=True)
class Graph2D:
    """Reusable 2D canvas graph output."""

    id: str
    title: UserString
    description: UserString
    x_label: UserString
    y_label: UserString
    series: tuple[GraphSeries, ...]
    interaction_channel: str | None = None

    def payload(self) -> dict[str, Any]:
        """Return JSON payload consumed by the browser graph renderer."""
        return {
            "id": self.id,
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "interaction_channel": self.interaction_channel,
            "series": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "points": [
                        {
                            "x": p.x,
                            "y": p.y,
                            "entity_id": p.entity_id,
                            "label": p.label,
                            "meta": dict(p.meta),
                        }
                        for p in s.points
                    ],
                }
                for s in self.series
            ],
        }


@dataclass(frozen=True, slots=True)
class MatrixCell:
    """One selectable matrix cell."""

    i: int
    j: int
    value: Any
    entity_id: str | None = None


@dataclass(frozen=True, slots=True)
class Matrix:
    """Rectangular matrix/heatmap-style diagnostic output."""

    id: str
    title: UserString
    description: UserString
    row_labels: tuple[UserString, ...]
    col_labels: tuple[UserString, ...]
    cells: tuple[MatrixCell, ...]
    interaction_channel: str | None = None


@dataclass(frozen=True, slots=True)
class WebGLView:
    """JSON-backed WebGL visualisation payload."""

    id: str
    title: UserString
    description: UserString
    renderer: Literal["region_surface", "graphene_viewer"]
    payload: dict[str, Any]
    interaction_channel: str | None = None


@dataclass(frozen=True, slots=True)
class EntityDetail:
    """Detailed fields displayed when a selectable entity is chosen."""

    entity_id: str
    title: UserString
    fields: tuple[tuple[str, Any], ...]


@dataclass(frozen=True, slots=True)
class DiagnosticSection:
    id: str
    title: UserString
    description: UserString = ""
    collapsed: bool = False
    body: tuple[DocumentBlock, ...] = ()
    markdowns: tuple[MarkdownBlock, ...] = ()
    math_blocks: tuple[TypstMathBlock, ...] = ()
    cards: tuple[Card, ...] = ()
    tables: tuple[Table, ...] = ()
    sections: tuple["DiagnosticSection", ...] = ()


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    """Complete output of one diagnostic run.

    It contains only structured data.  The renderer decides how to display it.
    """

    title: UserString
    summary: UserString
    cards: tuple[Card, ...] = ()
    tables: tuple[Table, ...] = ()
    graphs: tuple[Graph2D, ...] = ()
    matrices: tuple[Matrix, ...] = ()
    webgl: tuple[WebGLView, ...] = ()
    entity_details: tuple[EntityDetail, ...] = ()
    notes: tuple[UserString, ...] = ()


    markdowns: tuple[MarkdownBlock, ...] = ()
    sections: tuple[DiagnosticSection, ...] = ()
@dataclass(frozen=True, slots=True)
class DiagnosticSpec:
    """Registry entry for one diagnostic."""

    id: str
    group: str
    title: UserString
    description: UserString
    inputs: tuple[InputSpec, ...]
    compute: Callable[[Any, dict[str, Any]], DiagnosticResult]
    tier: Literal["instant", "cheap", "expensive"] = "cheap"


@dataclass(frozen=True, slots=True)
class DiagnosticSetup:
    """Serializable clipboard format for recreating a diagnostic setup."""

    version: int
    diagnostic_id: str
    inputs: dict[str, Any]
    view: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize setup to pretty JSON for clipboard use."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "DiagnosticSetup":
        """Parse setup JSON from the clipboard."""
        data = json.loads(text)
        return cls(
            version=int(data.get("version", 1)),
            diagnostic_id=str(data["diagnostic_id"]),
            inputs=dict(data.get("inputs", {})),
            view=dict(data.get("view", {})),
        )


def parse_inputs(spec: DiagnosticSpec, raw: Mapping[str, Any]) -> dict[str, Any]:
    """Parse inputs for a diagnostic from query params or JSON mapping."""
    return {inp.name: inp.parse(raw.get(inp.name)) for inp in spec.inputs}


def serialize_inputs(spec: DiagnosticSpec, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize parsed input values into stable JSON-compatible values."""
    return {inp.name: inp.serialize(inputs.get(inp.name, inp.default)) for inp in spec.inputs}
