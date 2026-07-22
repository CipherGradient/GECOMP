# Adding an Operator

This note explains how to extend the GECOMP perturbation library.

## Contract

Every operator must follow the same signature:

```python
def my_op(text: str, target: str = "", **kwargs) -> tuple[str, bool]:
    ...
```

- `text`: full input sentence
- `target`: the span/token selected by the pipeline (may be empty)
- return: `(new_text, ok)`
  - `ok=True` and `new_text != text` means a successful edit
  - otherwise the caller treats it as a no-op and may try another operator

Character-level, word-level, and other granularities are all expressed as text-in / text-out edits. The pipeline selects a target inside a visible window, then calls operators from `--ops_pool`.

## Steps

1. Copy `main/ops/_template_op.py` to a new file, e.g. `main/ops/my_op.py`.
2. Rename the function and change the register name:

```python
from main.ops.registry import register

@register("my_op")
def my_op(text: str, target: str = "", **_) -> tuple[str, bool]:
    ...
```

3. Import the module once so registration runs at startup. Add this line to `main/ops/__init__.py`:

```python
from . import my_op  # noqa: F401
```

4. Pass the name in the operator pool:

```bash
--ops_pool "misspell,homoglyph,semantic,phonetic,my_op"
```

