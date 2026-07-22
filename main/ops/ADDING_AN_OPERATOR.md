# Adding an Operator

This note explains how to extend the GenCOMP perturbation library without changing the training or attack core.

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

Character-level, word-level, and other granularities are all expressed as text-in / text-out edits. The pipeline selects a target inside a visible window, then calls one operator from `--ops_pool`.

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

That is enough for both training (`main.train`) and attack (`main.attack`). Candidate generation, filtering, and policy optimization consume the resulting text automatically.

## Built-in operators

| Name | Module | Level (informal) |
|------|--------|------------------|
| `misspell` | `char_misspell.py` | character |
| `homoglyph` | `char_homoglyph.py` | character |
| `semantic` | `word_semantic.py` | word |
| `phonetic` | `word_phonetic.py` | word |

List currently registered names:

```python
from main.ops.registry import available
print(available())
```

## Notes

- Pool entries are comma-separated names only (no per-op parameter syntax in the pool string). Extra parameters can be baked into your function defaults or read from kwargs if you call `apply_named` yourself.
- Unknown names in the pool are skipped.
- Keep edits local and reversible in spirit: prefer replacing `target` once rather than rewriting the whole sentence.
