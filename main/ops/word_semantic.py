import os
import sys
import warnings
from typing import Tuple, List, Dict, Optional
from collections import Counter
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['SPACY_WARNING_IGNORE'] = '1'

class _SuppressStderr:

    def __enter__(self):
        self._orig = sys.stderr
        try:
            self._null = open(os.devnull, 'w')
        except Exception:
            import io
            self._null = io.StringIO()
        sys.stderr = self._null
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            sys.stderr = self._orig
        finally:
            try:
                self._null.close()
            except Exception:
                pass
wn = None
_HAS_WORDNET = False
_WORDNET_ERROR: Optional[str] = None
try:
    from nltk.corpus import wordnet as wn
    _ = wn.synsets('dog')
    _HAS_WORDNET = True
except Exception as e:
    wn = None
    _HAS_WORDNET = False
    _WORDNET_ERROR = f'{type(e).__name__}: {e}'

def wordnet_ready() -> bool:
    return bool(_HAS_WORDNET and wn is not None)

def wordnet_error() -> Optional[str]:
    return _WORDNET_ERROR

def _require_wordnet() -> None:
    if not wordnet_ready():
        detail = _WORDNET_ERROR or 'unknown'
        raise RuntimeError(f'semantic op requires NLTK WordNet. Load failed: {detail}. Run: python -c "import nltk; nltk.download(\'wordnet\'); nltk.download(\'omw-1.4\')"')

def _get_semantic_candidates(word: str) -> List[str]:
    _require_wordnet()
    try:
        synset = wn.synsets(word)
    except Exception:
        return []
    if len(synset) == 0:
        return []
    posset = [syn.name().split('.')[1] for syn in synset]
    pos = Counter(posset).most_common(1)[0][0]
    lemmas: List[str] = []
    for syn in synset:
        if syn.name().split('.')[1] == pos:
            for l in syn.lemmas():
                name = l.name().replace('_', ' ')
                if name not in lemmas:
                    lemmas.append(name)
    return [w for w in lemmas if w.lower() != word.lower()]

def synonym_sub(text: str, target: str='movie', **_) -> Tuple[str, bool]:
    _require_wordnet()
    if not target or target not in text:
        return (text, False)
    cands = _get_semantic_candidates(target)
    if not cands:
        return (text, False)
    new_tok = cands[0]
    return (text.replace(target, new_tok, 1), True)
_MLM_PIPELINE = None

def _get_mlm_pipeline(model_name: str='bert-base-uncased'):
    global _MLM_PIPELINE
    if _MLM_PIPELINE is not None:
        return _MLM_PIPELINE
    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForMaskedLM.from_pretrained(model_name)
        _MLM_PIPELINE = pipeline('fill-mask', model=mdl, tokenizer=tok)
    except Exception:
        _MLM_PIPELINE = None
    return _MLM_PIPELINE

def mlm_word_sub(text: str, target: str='good', topk: int=20, **_) -> Tuple[str, bool]:
    if not target or target not in text:
        return (text, False)
    pipe = _get_mlm_pipeline()
    if pipe is None:
        return (text, False)
    mask_token = pipe.tokenizer.mask_token or '[MASK]'
    try:
        masked = text.replace(target, mask_token, 1)
        preds = pipe(masked, top_k=max(5, int(topk)))
        if preds and isinstance(preds[0], list):
            preds = preds[0]
        for p in preds:
            w = p.get('token_str') or p.get('sequence')
            if not isinstance(w, str):
                continue
            cand = text.replace(target, w.strip(), 1)
            if cand != text:
                return (cand, True)
    except Exception:
        return (text, False)
    return (text, False)
_NLP = None

def _get_spacy():
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        import spacy
        try:
            with _SuppressStderr():
                _NLP = spacy.load('en_core_web_sm')
        except Exception:
            _NLP = None
    except Exception:
        _NLP = None
    return _NLP

def _match_case(src: str, dst: str) -> str:
    if not src:
        return dst
    if src.isupper():
        return dst.upper()
    if src[0].isupper():
        return dst.capitalize()
    return dst.lower()

def _map_upos_to_wnpos(upos: str) -> Optional[str]:
    up = upos.upper()
    if up.startswith('NOUN') or up == 'NOUN':
        return 'n'
    if up.startswith('VERB') or up == 'VERB':
        return 'v'
    if up.startswith('ADJ') or up == 'ADJ':
        return 'a'
    if up.startswith('ADV') or up == 'ADV':
        return 'r'
    return None

def _wordnet_synonyms_for_pos(lemma: str, pos: Optional[str]) -> List[str]:
    if not _HAS_WORDNET or wn is None:
        return []
    try:
        synsets = wn.synsets(lemma, pos=pos) if pos else wn.synsets(lemma)
    except Exception:
        return []
    out: List[str] = []
    seen = set()
    for s in synsets:
        for l in s.lemmas():
            w = l.name().replace('_', ' ')
            wl = w.lower()
            if wl == lemma.lower() or wl in seen:
                continue
            seen.add(wl)
            out.append(w)
    return out

def wordnet_substitute_targets(text: str, targets: List[str], per_target: int=2) -> List[str]:
    results: List[str] = []
    nlp = _get_spacy()
    lower_targets = {t.lower() for t in targets}
    if nlp is not None and _HAS_WORDNET:
        try:
            import lemminflect
        except Exception:
            lemminflect = None
        with _SuppressStderr():
            doc = nlp(text)
        for tok in doc:
            if tok.text.lower() not in lower_targets:
                continue
            wn_pos = _map_upos_to_wnpos(tok.pos_)
            syns = _wordnet_synonyms_for_pos(tok.lemma_, wn_pos)
            if not syns:
                continue
            picked = syns[:max(1, per_target)]
            for sw in picked:
                rep = sw
                if 'lemminflect' in globals() or 'lemminflect' in locals():
                    try:
                        from lemminflect import getInflection
                        with _SuppressStderr():
                            infl = getInflection(sw, tag=tok.tag_)
                        if infl:
                            rep = infl[0]
                    except Exception:
                        pass
                rep = _match_case(tok.text, rep)
                (start, end) = (tok.idx, tok.idx + len(tok.text))
                cand = text[:start] + rep + text[end:]
                if cand != text:
                    results.append(cand)
    else:
        for t in targets:
            syns = _get_semantic_candidates(t)
            if not syns:
                continue
            for sw in syns[:max(1, per_target)]:
                cand = text.replace(t, _match_case(t, sw), 1)
                if cand != text:
                    results.append(cand)
    (uniq, seen) = ([], set())
    for s in results:
        k = s.strip()
        if k and k not in seen and (k != text):
            seen.add(k)
            uniq.append(k)
    return uniq
