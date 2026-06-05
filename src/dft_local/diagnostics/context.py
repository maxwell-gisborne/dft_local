"""Runtime context and cache for diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal
import json

from dft_local.core.dataset import SparseDataset
from dft_local.core.geometry import (
    EdgeDirections,
    EdgeGroupLabels,
    NearestNeighbourGraph,
)
from dft_local.core.kernels import GdKernelArrays


KernelChoice = Literal["anchored", "average", "average_star", "anchored_star"]


@dataclass
class DiagnosticsCache:
    """Small in-memory cache keyed by diagnostic id and serialised inputs."""

    values: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    max_items: int = 64

    def get_or_compute(self, key: tuple[Any, ...], build: Callable[[], Any]) -> Any:
        """Return cached value, computing it if missing."""

        if key in self.values:
            return self.values[key]

        value = build()

        if len(self.values) >= self.max_items:
            first = next(iter(self.values))
            del self.values[first]

        self.values[key] = value
        return value

    def clear(self) -> None:
        """Clear all cached computations."""

        self.values.clear()


@dataclass(frozen=True)
class DiagnosticsState:
    """Loaded physical state used by diagnostics."""

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
    KH_star: GdKernelArrays
    KS_star: GdKernelArrays

    @classmethod
    def from_root(cls, root: str | Path) -> "DiagnosticsState":
        """Load all long-lived data required by diagnostics."""

        data = SparseDataset.load(Path(root))
        geom = NearestNeighbourGraph.from_positions(data.metadata.positions)
        edges = EdgeDirections.from_geometry(geom)
        labels = EdgeGroupLabels.from_geometry(geom, edges)

        KH = GdKernelArrays.from_anchored(data.H, labels, matrix_name="H anchored")
        KS = GdKernelArrays.from_anchored(data.S, labels, matrix_name="S anchored")
        KH_avg = GdKernelArrays.from_average(data.H, labels, matrix_name="H average")
        KS_avg = GdKernelArrays.from_average(data.S, labels, matrix_name="S average")
        KH_star = KH.star_symmetrised(matrix_name="H anchored star")
        KS_star = KS.star_symmetrised(matrix_name="S anchored star")
        KH_avg_star = KH_avg.star_symmetrised(matrix_name="H average star")
        KS_avg_star = KS_avg.star_symmetrised(matrix_name="S average star")

        return cls(
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
            KH_star=KH_star,
            KS_star=KS_star,
        )

    @property
    def units(self):
        """Return dataset units."""

        return self.data.units

    def unit_provenance_rows(self) -> list[list[object]]:
        """Rows describing disk-to-working unit provenance for the loaded dataset."""

        return [
            ["disk energy unit", self.data.disk_unit_context.energy.symbol],
            ["working energy unit", self.data.working_unit_context.energy.symbol],
            ["disk length unit", self.data.disk_unit_context.length.symbol],
            ["working length unit", self.data.working_unit_context.length.symbol],
            ["energy disk-to-working factor", self.data.energy_conversion_disk_to_working],
            ["length disk-to-working factor", self.data.length_conversion_disk_to_working],
        ]

    def kernels(self, choice: KernelChoice) -> tuple[GdKernelArrays, GdKernelArrays]:
        """Return ``(KH, KS)`` for a named kernel variant."""

        match choice:
            case "anchored":
                return self.KH, self.KS
            case "anchored_star":
                return self.KH_star, self.KS_star
            case "average":
                return self.KH_avg, self.KS_avg
            case "average_star":
                return self.KH_avg_star, self.KS_avg_star
            case _:
                raise ValueError(f"Unknown kernel choice: {choice}")


@dataclass
class DiagnosticContext:
    """Context passed to each diagnostic compute function."""

    state: DiagnosticsState
    cache: DiagnosticsCache = field(default_factory=DiagnosticsCache)

    def kernels(self, choice: KernelChoice) -> tuple[GdKernelArrays, GdKernelArrays]:
        """Forward kernel selection to loaded state."""

        return self.state.kernels(choice)

    def cached(self, diagnostic_id: str, inputs: dict[str, Any], build: Callable[[], Any]) -> Any:
        """Cache a diagnostic computation by id and JSON-serialised inputs."""

        key = (diagnostic_id, json.dumps(inputs, sort_keys=True, default=str))
        return self.cache.get_or_compute(key, build)
