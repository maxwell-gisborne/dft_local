"""User-facing strings with optional rich scientific rendering.

Plain ``str`` remains the default.  ``TypstMath`` is an opt-in rich string
for places that already accept user-visible text, such as titles, labels,
descriptions, notes, and table headers.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class RichText:
    """String-like user text made from plain text and rich inline parts."""

    parts: tuple["UserString", ...]


@dataclass(frozen=True, slots=True)
class TypstMath:
    """Typst math snippet used anywhere a user-facing string may appear.

    ``source`` should normally include Typst math delimiters, for example
    ``$ H(k) u = E S(k) u $``.

    ``name`` is optional but useful in tests because it gives compile failures
    a stable human-readable location.
    """

    source: str
    display: bool = False
    name: str = ""


UserString: TypeAlias = str | TypstMath | RichText


def iter_typst_math(obj: object) -> Iterator[TypstMath]:
    """Yield every TypstMath object reachable from a structured object.

    This is intentionally independent of rendering.  Tests can walk diagnostic
    specs/results and compile every authored Typst snippet.
    """

    seen: set[int] = set()

    def walk(value: object) -> Iterator[TypstMath]:
        ident = id(value)
        if ident in seen:
            return
        seen.add(ident)

        if isinstance(value, TypstMath):
            yield value
            return

        if isinstance(value, (str, bytes, bytearray, int, float, complex, bool, type(None))):
            return

        if is_dataclass(value):
            for field in fields(value):
                yield from walk(getattr(value, field.name))
            return

        if isinstance(value, Mapping):
            for key, item in value.items():
                yield from walk(key)
                yield from walk(item)
            return

        if isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                yield from walk(item)
            return

    yield from walk(obj)



def math(source: str, *, display: bool = False, name: str = "") -> TypstMath:
    """Create a Typst math user string."""

    return TypstMath(source, display=display, name=name)


def rich(*parts: UserString) -> RichText:
    """Create mixed plain-text / rich-text user string content."""

    return RichText(tuple(parts))
