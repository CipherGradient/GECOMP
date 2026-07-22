from typing import Tuple
import random
LETTER_MAPPINGS = {'a': ['а', 'ạ', 'ȧ', 'ḁ', 'ā', 'ą', 'ä', 'ӓ', 'ã', 'á', 'â', 'à', 'å'], 'b': ['ḅ', 'þ', 'ɓ'], 'c': ['ᴄ', 'с', 'ⅽ', 'ϲ', 'ç', 'ć'], 'd': ['ⅾ', 'ḍ', 'ɖ', 'đ'], 'e': ['е', 'ẹ', 'ė', 'ē', 'ę', 'ë', 'ё', 'è', 'ê', 'é', 'ℯ', 'ɛ'], 'f': ['ƒ', 'ḟ'], 'g': ['ɡ', 'ġ', 'ĝ', 'ğ'], 'h': ['һ', 'ḥ', 'ħ'], 'i': ['і', 'ị', 'į', 'ı', 'ι'], 'j': ['ј', 'ȷ', 'ĵ'], 'k': ['ķ', 'ḳ', 'κ', 'к'], 'l': ['ⅼ', 'ļ', 'ł'], 'm': ['ⅿ', 'ṃ', 'ṁ'], 'n': ['ո', 'ņ', 'ṅ', 'η', 'ñ'], 'o': ['ο', 'о', 'ọ', 'ö', 'õ', 'ó', 'ô', 'ò', 'ø', 'ɵ'], 'p': ['р', 'ρ', 'ṗ'], 'q': ['ԛ'], 'r': ['ŗ', 'ṟ', 'г', 'ř'], 's': ['ѕ', 'ș', 'ṣ', 'ś', 'š'], 't': ['ț', 'ṭ', 'ţ', 'ŧ', 'ť'], 'u': ['ս', 'ụ', 'ū', 'ü', 'ù', 'ú', 'û'], 'v': ['ᴠ', 'ṿ', 'ѵ'], 'w': ['ᴡ', 'ẉ', 'ẇ', 'ẅ'], 'x': ['х', 'ẋ', 'ӿ'], 'y': ['у', 'ỵ', 'ÿ', 'ý', 'ŷ'], 'z': ['ᴢ', 'ẓ', 'ż', 'ž']}

def _swap_chars(token: str) -> str:
    out = []
    for ch in token:
        repls = LETTER_MAPPINGS.get(ch, LETTER_MAPPINGS.get(ch.lower()))
        if repls:
            out.append(random.choice(repls))
        else:
            out.append(ch)
    return ''.join(out)

def homoglyph_swap(text: str, target: str='good', **_) -> Tuple[str, bool]:
    if not target or target not in text:
        return (text, False)
    return (text.replace(target, _swap_chars(target), 1), True)

def homoglyph_sentence(text: str, perturb_pct: float=0.25, seed: int=0, **_) -> Tuple[str, bool]:
    if not text:
        return (text, False)
    random.seed(seed)
    chars = list(text)
    num = max(1, int(len(chars) * max(0.0, min(1.0, perturb_pct))))
    idxs = list(range(len(chars)))
    random.shuffle(idxs)
    changed = False
    for i in idxs[:num]:
        ch = chars[i]
        repls = LETTER_MAPPINGS.get(ch, LETTER_MAPPINGS.get(ch.lower()))
        if repls:
            chars[i] = random.choice(repls)
            changed = True
    new_text = ''.join(chars)
    return (new_text, changed)
_ZW = ['\u200b', '\u200c', '\u200d', '\ufeff']

def zerowidth_insert(text: str, pos: int=0, count: int=1, **_) -> Tuple[str, bool]:
    if not text:
        return (text, False)
    pos = max(0, min(len(text), int(pos)))
    n = max(1, int(count))
    zw = _ZW[0] * n
    return (text[:pos] + zw + text[pos:], True)
