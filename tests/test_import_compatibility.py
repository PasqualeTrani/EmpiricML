"""Names and signatures that user code may import must survive refactoring.

The expected signatures were recorded before the Lab refactor. Helper
functions are not documented, but they stay importable from their original
modules for backward compatibility.
"""

import importlib
import inspect
import json
from pathlib import Path

import pytest

SIGNATURES = json.loads(
    (Path(__file__).parent / "fixtures" / "public_signatures.json").read_text()
)


def _resolve(dotted: str):
    parts = dotted.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for attr in parts[split:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(dotted)


@pytest.mark.parametrize("dotted", sorted(SIGNATURES))
def test_public_name_keeps_signature(dotted):
    obj = _resolve(dotted)
    target = obj.__init__ if inspect.isclass(obj) else inspect.unwrap(obj)
    assert str(inspect.signature(target)) == SIGNATURES[dotted]
