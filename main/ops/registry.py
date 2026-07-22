from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
import random
import warnings

OpFn = Callable[..., Tuple[str, bool]]
_REGISTRY: Dict[str, OpFn] = {}
_BUILTINS_LOADED = False
_BUILTIN_NAMES = ('misspell', 'homoglyph', 'semantic', 'phonetic')

def register(name: str, fn: Optional[OpFn]=None):
    key = str(name).strip().lower()

    def _wrap(f: OpFn) -> OpFn:
        if not key:
            raise ValueError('empty op name')
        _REGISTRY[key] = f
        return f
    if fn is not None:
        return _wrap(fn)
    return _wrap

def _warn_reg(name: str, err: BaseException) -> None:
    msg = f'[ops] failed to register builtin "{name}": {type(err).__name__}: {err}'
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    print(msg)

def ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    try:
        from .char_misspell import apply_keyboard_typo
        register('misspell', apply_keyboard_typo)
    except Exception as e:
        _warn_reg('misspell', e)
    try:
        from .char_homoglyph import homoglyph_swap
        register('homoglyph', homoglyph_swap)
    except Exception as e:
        _warn_reg('homoglyph', e)
    try:
        from .word_semantic import synonym_sub, wordnet_ready
        if not wordnet_ready():
            raise RuntimeError('NLTK WordNet unavailable (install nltk and download wordnet/omw-1.4)')
        register('semantic', synonym_sub)
    except Exception as e:
        _warn_reg('semantic', e)
    try:
        from .word_phonetic import phonetic_word_sub, wild_ready, wild_error
        if not wild_ready():
            detail = wild_error() or 'WILD/spacy load failed'
            raise RuntimeError(f'phonetic backend unavailable: {detail}')
        register('phonetic', phonetic_word_sub)
    except Exception as e:
        _warn_reg('phonetic', e)
    missing = [n for n in _BUILTIN_NAMES if n not in _REGISTRY]
    if missing:
        msg = f'[ops] builtins missing after registration: {missing}; available={sorted(_REGISTRY.keys())}'
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        print(msg)

def get(name: str) -> Optional[OpFn]:
    ensure_builtins()
    return _REGISTRY.get(str(name).strip().lower())

def available() -> List[str]:
    ensure_builtins()
    return sorted(_REGISTRY.keys())

def parse_pool(pool: str) -> List[str]:
    return [p.strip() for p in str(pool or '').split(',') if p.strip()]

def apply_named(name: str, text: str, *, target: str='', **kwargs) -> Tuple[str, bool]:
    key = str(name).strip().lower()
    fn = get(key)
    if fn is None:
        raise RuntimeError(f'operator "{key}" is not registered; available={available()}')
    try:
        (y2, ok) = fn(text, target=target, **kwargs)
        return (y2, bool(ok))
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f'operator "{key}" failed: {type(e).__name__}: {e}') from e

def apply_one_from_pool(text: str, pool: str, *, target: str, rng=None) -> Tuple[str, bool]:
    r = rng or random
    names = parse_pool(pool)
    if not names or not target:
        return (text, False)
    ensure_builtins()
    missing = [n for n in names if get(n) is None]
    if missing:
        raise RuntimeError(f'ops_pool references unregistered operators {missing}; available={available()}. Check nltk/spacy/WILD deps.')
    choices = names[:]
    r.shuffle(choices)
    for name in choices:
        (y2, ok) = apply_named(name, text, target=target)
        if ok and y2 != text:
            return (y2, True)
    return (text, False)
