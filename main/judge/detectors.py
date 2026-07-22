from __future__ import annotations
from typing import List, Tuple, Optional
import re
try:
    from spellchecker import SpellChecker
except Exception:
    SpellChecker = None
try:
    import homoglyphs as hg
except Exception:
    hg = None

def _proxy_score(text: str) -> float:
    unk_rate = 0.0
    tokens = re.findall("[A-Za-z']+", text)
    if SpellChecker is not None and len(tokens) > 0:
        try:
            sp = SpellChecker(language='en')
            unknown = sp.unknown([t.lower() for t in tokens])
            unk_rate = min(1.0, len(unknown) / max(1, len(tokens)))
        except Exception:
            unk_rate = 0.0
    glyph_frac = 0.0
    if hg is not None and len(text) > 0:
        try:
            H = hg.Homoglyphs(languages={'en'}, strategy=hg.STRATEGY_LOAD)
            mapped = []
            for ch in text:
                try:
                    ascii_cand = H.to_ascii(ch)[0]
                except Exception:
                    ascii_cand = ch
                mapped.append(1 if ascii_cand != ch else 0)
            glyph_frac = sum(mapped) / len(text)
        except Exception:
            glyph_frac = 0.0
    else:
        non_ascii = sum((1 for c in text if ord(c) > 127))
        glyph_frac = non_ascii / max(1, len(text))
    rep = 1.0 if re.search('([!?.,;:])\\1{2,}', text) or re.search('\\s{3,}', text) else 0.0
    return max(unk_rate, glyph_frac, rep)

def proxy_scores(texts: List[str]) -> List[float]:
    return [_proxy_score(t) for t in texts]

def calibrate_proxy_threshold(clean_texts: List[str], target_fpr: float=0.05) -> float:
    import numpy as np
    scores = np.array(proxy_scores(clean_texts), dtype=float)
    q = float(np.quantile(scores, 1.0 - target_fpr))
    return q

def detect_with_proxy(texts: List[str], threshold: float) -> Tuple[float, List[bool]]:
    scores = proxy_scores(texts)
    flags = [s > threshold for s in scores]
    rate = sum(flags) / max(1, len(flags))
    return (rate, flags)
