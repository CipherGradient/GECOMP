from __future__ import annotations
from typing import Tuple
from main.ops.registry import register


@register("my_op")
def my_op(text: str, target: str = "", **_) -> Tuple[str, bool]:
    if not target or target not in text:
        return text, False
    new_token = target[::-1]
    if new_token == target:
        return text, False
    return text.replace(target, new_token, 1), True
