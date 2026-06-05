"""Small helpers for constructing ordered diagnostic documents."""

from __future__ import annotations

from dft_local.diagnostics.models import EquationBlock, ProseBlock
from dft_local.diagnostics.user_strings import TypstMath, UserString


def prose(id: str, title: UserString, markdown: UserString) -> ProseBlock:
    return ProseBlock(id=id, title=title, markdown=markdown)


def equation(id: str, source: str, *, name: str = "") -> EquationBlock:
    return EquationBlock(
        id=id,
        math=TypstMath(source, display=True, name=name or id),
    )
