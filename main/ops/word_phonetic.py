from typing import Tuple, List, Optional
import os
_WILD = None
_WILD_ERROR: Optional[str] = None
try:
    from main.vendor.wild_process import WILD
    _WILD = WILD()
    vendor_wild_dir = os.path.join(os.path.dirname(__file__), '..', 'vendor', 'wild')
    vendor_wild_dir = os.path.normpath(vendor_wild_dir)
    if os.path.isdir(vendor_wild_dir):
        _WILD.load(vendor_wild_dir)
    else:
        _WILD.load('./wild')
except Exception as e:
    _WILD = None
    _WILD_ERROR = f'{type(e).__name__}: {e}'

def wild_ready() -> bool:
    return _WILD is not None

def wild_error() -> Optional[str]:
    return _WILD_ERROR

def _require_wild() -> None:
    if _WILD is None:
        detail = _WILD_ERROR or 'unknown load failure'
        raise RuntimeError(f'phonetic op requires WILD (+ spacy). Load failed: {detail}. Ensure main/vendor/wild exists and spacy is installed.')

def _get_phonetic_candidates(word: str, level: int=1, distance: int=5, strict: bool=True) -> List[str]:
    _require_wild()
    try:
        sims = _WILD.get_similars(word, level=level, distance=distance, strict=strict)
    except Exception:
        sims = []
    return [w for w in list(sims) if w.lower() != word.lower()]

def phonetic_word_sub(text: str, target: str='movie', level: int=1, distance: int=5, strict: bool=True, **_) -> Tuple[str, bool]:
    _require_wild()
    if not target or target not in text:
        return (text, False)
    cands = _get_phonetic_candidates(target, level=level, distance=distance, strict=strict)
    if not cands:
        return (text, False)
    new_tok = cands[0]
    return (text.replace(target, new_tok, 1), True)
