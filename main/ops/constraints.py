from __future__ import annotations
import re
from typing import Optional, Tuple, Dict
import math
import torch
_SBERT: Optional[object] = None
_SBERT_TRIED: bool = False
_GPT2_MODEL: Optional[object] = None
_GPT2_TOK: Optional[object] = None
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_SPELL: Optional[object] = None
_HOMO: Optional[object] = None

def _parse_spec(spec: Optional[str], default_op: str, default_val: float) -> Tuple[str, float]:
    if not spec:
        return (default_op, default_val)
    m = re.search('([<>]=?)\\s*([0-9]*\\.?[0-9]+)', str(spec))
    if not m:
        return (default_op, default_val)
    return (m.group(1), float(m.group(2)))

def _cmp(op: str, lhs: float, rhs: float) -> bool:
    if op == '>=':
        return lhs >= rhs
    if op == '>':
        return lhs > rhs
    if op == '<=':
        return lhs <= rhs
    if op == '<':
        return lhs < rhs
    return True

def _ensure_sbert():
    global _SBERT, _SBERT_TRIED
    if _SBERT_TRIED:
        return
    _SBERT_TRIED = True
    try:
        from pathlib import Path
        from sentence_transformers import SentenceTransformer
        device_str = 'cuda' if _DEVICE.type == 'cuda' else 'cpu'
        here = Path(__file__).resolve()
        gecomp_model = here.parents[2] / 'model'
        local_dirs = [gecomp_model / 'sentence-transformers__paraphrase-MiniLM-L6-v2', Path('model/sentence-transformers__paraphrase-MiniLM-L6-v2')]
        last_err: Optional[BaseException] = None
        for path in local_dirs:
            if not path.exists():
                continue
            try:
                _SBERT = SentenceTransformer(str(path), device=device_str)
                print(f'[init] SentenceTransformer MiniLM loaded from {path}')
                return
            except Exception as e:
                last_err = e
                _SBERT = None
        try:
            _SBERT = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2', device=device_str)
            print('[init] SentenceTransformer MiniLM loaded from Hub id')
            return
        except Exception as e:
            last_err = e
            _SBERT = None
        print(f'[WARN] SentenceTransformer MiniLM unavailable ({type(last_err).__name__}: {last_err}); SIM gates will be skipped. Place model/sentence-transformers__paraphrase-MiniLM-L6-v2 under gecomp/model.')
    except Exception as e:
        _SBERT = None
        print(f'[WARN] SentenceTransformer import/load failed ({type(e).__name__}: {e})')

def passes_similarity(src: str, dst: str, spec: str) -> bool:
    (op, thr) = _parse_spec(spec, '>=', 0.88)
    _ensure_sbert()
    if _SBERT is None:
        return True
    try:
        from sentence_transformers import util
        emb = _SBERT.encode([src, dst], convert_to_tensor=True)
        sim = float(util.cos_sim(emb[0], emb[1]).item())
        return _cmp(op, sim, thr)
    except Exception:
        return True
_GPT2_LOAD_ERROR: Optional[str] = None
_GPT2_WARNED: bool = False
_GPT2_FORCE_CPU: bool = False

def prefer_gpt2_cpu(enabled: bool=True) -> None:
    global _GPT2_FORCE_CPU, _GPT2_MODEL, _GPT2_TOK
    _GPT2_FORCE_CPU = bool(enabled)
    if not enabled:
        return
    if _GPT2_MODEL is not None:
        try:
            _GPT2_MODEL.to(torch.device('cpu'))
            _GPT2_MODEL.eval()
            print('[init] GPT-2 PPL pinned to CPU (avoid mid-train CUDA→CPU cliff).')
        except Exception:
            pass

def _ensure_gpt2():
    global _GPT2_MODEL, _GPT2_TOK, _GPT2_LOAD_ERROR, _GPT2_WARNED
    if _GPT2_MODEL is not None and _GPT2_TOK is not None:
        return
    try:
        from pathlib import Path as _Path
        from transformers import AutoModelForCausalLM, AutoTokenizer
        here = _Path(__file__).resolve()
        local_gpt2 = here.parents[2] / 'model' / 'gpt2'
        gpt2_id = str(local_gpt2) if local_gpt2.exists() else 'gpt2'
        _GPT2_TOK = AutoTokenizer.from_pretrained(gpt2_id, trust_remote_code=True)
        if getattr(_GPT2_TOK, 'pad_token', None) is None:
            _GPT2_TOK.pad_token = _GPT2_TOK.eos_token
        model = None
        last_err: Optional[BaseException] = None
        use_cuda = _DEVICE.type == 'cuda' and (not _GPT2_FORCE_CPU)
        if use_cuda:
            try:
                model = AutoModelForCausalLM.from_pretrained(gpt2_id, torch_dtype=torch.float16, trust_remote_code=True)
                model.to(_DEVICE)
            except Exception as e:
                last_err = e
                model = None
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(gpt2_id, trust_remote_code=True)
            model.to(torch.device('cpu'))
            if _GPT2_FORCE_CPU and (not _GPT2_WARNED):
                print('[init] GPT-2 PPL loaded on CPU (forced for stable training).')
                _GPT2_WARNED = True
            elif last_err is not None and (not _GPT2_WARNED):
                print(f'[WARN] GPT-2 PPL on CUDA failed ({type(last_err).__name__}: {last_err}); falling back to CPU.')
                _GPT2_WARNED = True
        model.eval()
        _GPT2_MODEL = model
        _GPT2_LOAD_ERROR = None
    except Exception as e:
        _GPT2_MODEL = None
        _GPT2_TOK = None
        _GPT2_LOAD_ERROR = f'{type(e).__name__}: {e}'
        if not _GPT2_WARNED:
            print(f'[WARN] GPT-2 PPL unavailable ({_GPT2_LOAD_ERROR}); quality_gate will skip PPL checks.')
            _GPT2_WARNED = True

def _ppl_forward(text: str) -> Optional[float]:
    if _GPT2_MODEL is None or _GPT2_TOK is None:
        return None
    model_device = next(_GPT2_MODEL.parameters()).device
    inputs = _GPT2_TOK(text, return_tensors='pt', truncation=True, max_length=256)
    inputs = {k: v.to(model_device) for (k, v) in inputs.items()}
    with torch.no_grad():
        if model_device.type == 'cuda':
            with torch.autocast(device_type='cuda', enabled=False):
                loss = _GPT2_MODEL(**inputs, labels=inputs['input_ids']).loss
        else:
            loss = _GPT2_MODEL(**inputs, labels=inputs['input_ids']).loss
    loss_val = float(loss.item())
    if not math.isfinite(loss_val):
        return None
    return float(math.exp(min(loss_val, 14.0)))

def _ppl(text: str) -> Optional[float]:
    global _GPT2_WARNED
    _ensure_gpt2()
    if _GPT2_MODEL is None or _GPT2_TOK is None:
        return None
    try:
        ppl = _ppl_forward(text)
        if ppl is not None:
            return ppl
        return None
    except Exception as e:
        try:
            if _DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception:
            pass
        if not _GPT2_WARNED:
            print(f'[WARN] GPT-2 PPL forward failed ({type(e).__name__}: {e}); skipping this PPL (model stays on its original device).')
            _GPT2_WARNED = True
        return None

def passes_ppl_delta(src: str, dst: str, spec: str) -> bool:
    (op, thr) = _parse_spec(spec, '<=', 1.2)
    ppl_src = _ppl(src)
    ppl_dst = _ppl(dst)
    if ppl_src is None or ppl_dst is None:
        return True
    ratio = ppl_dst / max(1e-08, ppl_src)
    return _cmp(op, ratio, thr)

def _ensure_detectors():
    global _SPELL, _HOMO
    if _SPELL is None:
        try:
            from spellchecker import SpellChecker
            _SPELL = SpellChecker(language='en')
        except Exception:
            _SPELL = None
    if _HOMO is None:
        try:
            import homoglyphs as hg
            _HOMO = hg.Homoglyphs(languages={'en'}, strategy=hg.STRATEGY_LOAD)
        except Exception:
            _HOMO = None

def _detector_score(dst: str) -> float:
    _ensure_detectors()
    tokens = re.findall("[A-Za-z']+", dst)
    unk_rate = 0.0
    if _SPELL is not None and len(tokens) > 0:
        unknown = _SPELL.unknown([t.lower() for t in tokens])
        unk_rate = min(1.0, len(unknown) / max(1, len(tokens)))
    glyph_frac = 0.0
    if _HOMO is not None and len(dst) > 0:
        try:
            mapped = []
            for ch in dst:
                try:
                    ascii_cand = _HOMO.to_ascii(ch)[0]
                except Exception:
                    ascii_cand = ch
                mapped.append(1 if ascii_cand != ch else 0)
            glyph_frac = sum(mapped) / len(dst)
        except Exception:
            glyph_frac = 0.0
    else:
        non_ascii = sum((1 for c in dst if ord(c) > 127))
        glyph_frac = non_ascii / max(1, len(dst))
    rep = 0.0
    m1 = re.search('([!?.,;:])\\1{2,}', dst)
    m2 = re.search('\\s{3,}', dst)
    rep = 1.0 if m1 or m2 else 0.0
    return max(unk_rate, glyph_frac, rep)

def passes_detector(dst: str, spec: str) -> bool:
    (op, thr) = _parse_spec(spec, '<=', 0.5)
    score = _detector_score(dst)
    return _cmp(op, score, thr)

def compute_similarity(src: str, dst: str) -> float:
    _ensure_sbert()
    if _SBERT is None:
        return 1.0
    try:
        from sentence_transformers import util
        emb = _SBERT.encode([src, dst], convert_to_tensor=True)
        return float(util.cos_sim(emb[0], emb[1]).item())
    except Exception:
        return 1.0

def compute_ppl(text: str) -> Optional[float]:
    return _ppl(text)

def compute_ppl_ratio(src: str, dst: str, *, src_ppl: Optional[float]=None) -> float:
    p0 = float(src_ppl) if src_ppl is not None and math.isfinite(float(src_ppl)) else _ppl(src)
    p1 = _ppl(dst)
    if p0 is None or p1 is None:
        return 1.0
    return float(p1 / max(1e-08, p0))

def proxy_score(dst: str) -> float:
    return float(_detector_score(dst))

def normalized_levenshtein(a: str, b: str) -> float:
    try:
        import Levenshtein
        dist = Levenshtein.distance(a, b)
    except Exception:
        import difflib
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        max_len = max(len(a), len(b))
        dist = int(round((1.0 - ratio) * max_len))
    denom = max(1, max(len(a), len(b)))
    return dist / denom

def normalized_word_edit(a: str, b: str) -> float:
    try:
        import difflib
        ta = a.split()
        tb = b.split()
        sm = difflib.SequenceMatcher(a=ta, b=tb)
        edits = 0
        for (tag, i1, i2, j1, j2) in sm.get_opcodes():
            if tag == 'equal':
                continue
            if tag == 'replace':
                edits += max(i2 - i1, j2 - j1)
            elif tag == 'delete':
                edits += i2 - i1
            elif tag == 'insert':
                edits += j2 - j1
        return edits / max(1, len(ta))
    except Exception:
        return 0.0

def passes_edit_distance(src: str, dst: str, max_ratio: float=0.15) -> bool:
    return normalized_levenshtein(src, dst) <= max_ratio

def _token_count(text: str, tokenizer: Optional[object]=None) -> int:
    if tokenizer is None:
        return len(text.split())
    try:
        out = tokenizer(text, return_tensors='pt', truncation=True)
        return int(out['input_ids'].shape[-1])
    except Exception:
        return len(text.split())

def passes_tokenizer_shift(src: str, dst: str, max_ratio: float=0.25, tokenizer: Optional[object]=None) -> bool:
    n_src = _token_count(src, tokenizer)
    n_dst = _token_count(dst, tokenizer)
    ratio = abs(n_dst / max(1, n_src) - 1.0)
    return ratio <= max_ratio

def passes_budget(edits_so_far: int, max_edits: int) -> bool:
    return edits_so_far < max_edits

def passes_type_budget(op_counts: Dict[str, int], type_quotas: Dict[str, int]) -> bool:
    for (k, v) in type_quotas.items():
        if op_counts.get(k, 0) > v:
            return False
    return True
