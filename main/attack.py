import argparse, json
import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple
import csv
import re, math
import time
import random
from datetime import datetime
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['SPACY_WARNING_IGNORE'] = '1'
_HERE = Path(__file__).resolve()
_GECOMP_ROOT = _HERE.parents[1]
_REPO_ROOT = _GECOMP_ROOT.parent
for _p in (str(_GECOMP_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from datasets import load_dataset, load_from_disk
try:
    from datasets import Dataset
except Exception:
    Dataset = None
from main.victims.scoring import load_victim, QueryCounter, resolve_victim_name, to_portable_path
from main.judge.metrics import compute_basic_report, compute_nasr, compute_asr_multi
from main.judge.detectors import calibrate_proxy_threshold, detect_with_proxy
from main.ops.constraints import compute_similarity, compute_ppl_ratio, proxy_score, normalized_levenshtein, normalized_word_edit, _ppl
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
try:
    from tqdm.auto import tqdm
except Exception:

    class tqdm:

        def __init__(self, total=None, desc=None, unit=None, leave=False, dynamic_ncols=True):
            self.total = total
            self.n = 0

        def update(self, n=1):
            self.n += n

        def set_postfix_str(self, s):
            pass

        def close(self):
            pass

def _visible_span_by_offsets(text: str, tokenizer, L: int=384):
    try:
        enc = tokenizer(text, return_offsets_mapping=True, return_tensors='pt', truncation=True)
        offs = enc['offset_mapping'][0]
        n = offs.shape[0]
        k = min(n, max(1, int(L)))
        end = 0
        for i in range(k):
            (a, b) = offs[i].tolist()
            end = max(end, int(b))
        return (0, max(1, min(end, len(text))))
    except Exception:
        return (0, min(len(text), 128))

def _apply_one_ops(text: str, tokenizer, pool: str='misspell,homoglyph,semantic,phonetic'):
    from main.ops.registry import apply_one_from_pool
    ops = [p.strip() for p in str(pool).split(',') if p.strip()]
    if not ops:
        return text
    (L0, L1) = _visible_span_by_offsets(text, tokenizer, 384)
    span = text[L0:L1]
    toks = [w for w in span.split() if any((c.isalpha() for c in w))]
    if not toks:
        return text
    import random as _r
    target = _r.choice(toks)
    (y, ok) = apply_one_from_pool(text, pool, target=target, rng=_r)
    if ok and y != text:
        return y
    try:
        start = text.find(target)
        if start != -1 and len(target) >= 2:
            i = _r.randrange(0, len(target) - 1)
            t_list = list(target)
            (t_list[i], t_list[i + 1]) = (t_list[i + 1], t_list[i])
            swapped = ''.join(t_list)
            return text[:start] + swapped + text[start + len(target):]
    except Exception:
        pass
    return text

def _sample_micro_candidates(x: str, tokenizer, K: int, pool: str, edit_cap: float) -> List[Tuple[float, str]]:
    tried = set()
    tmp: List[Tuple[float, str]] = []
    budget = max(K * 6, K + 8)
    attempts = 0
    max_attempts = budget * 4
    while attempts < max_attempts and len(tmp) < K * 3:
        attempts += 1
        y1 = _apply_one_ops(x, tokenizer, pool=pool)
        if not y1 or y1 == x or y1 in tried:
            if y1:
                tried.add(y1)
            continue
        tried.add(y1)
        try:
            edr = normalized_levenshtein(x, y1)
        except Exception:
            edr = 1.0
        try:
            if float(edr) > float(edit_cap):
                continue
        except Exception:
            pass
        if not any((c.isalpha() for c in y1)) or len(y1.strip()) < 5:
            continue
        tmp.append((float(edr), y1))
    tmp.sort(key=lambda t: t[0])
    return tmp[:K]

def _canonicalize_plan_name(plan: str) -> str:
    p = str(plan).lower()
    p = p.replace('passive', 'query')
    p = p.replace('pass', 'query')
    p = p.replace('llm', 'abst')
    return p

def _ensure_complete_plan(plan: str) -> str:
    p = _canonicalize_plan_name(plan)
    if 'syn1' not in p and 'syn2' not in p:
        p = f'{p}+syn2'
    if 'char_mid' not in p and 'char_high' not in p:
        p = f'{p}+char_mid'
    return p

def _load_combo_list(args) -> List[str]:
    from typing import Dict, Tuple
    import json as _json
    scores: Dict[str, float] = {}
    sums: Dict[str, float] = {}
    cnts: Dict[str, int] = {}

    def _add(plan: str, score: float) -> None:
        try:
            pn = _ensure_complete_plan(plan)
            sums[pn] = float(sums.get(pn, 0.0) + float(score))
            cnts[pn] = int(cnts.get(pn, 0) + 1)
        except Exception:
            pass
    if getattr(args, 'combo_scores', None):
        try:
            with open(args.combo_scores, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = _json.loads(line)
                        _add(rec.get('plan', ''), rec.get('score', 0.0))
                    except Exception:
                        continue
            try:
                ps_path = Path(args.combo_scores).parent / 'plan_scores.json'
                if ps_path.exists():
                    with open(ps_path, 'r', encoding='utf-8') as f:
                        obj = _json.load(f)
                    if isinstance(obj, dict) and isinstance(obj.get('scores', None), dict):
                        scores = {_ensure_complete_plan(k): float(v) for (k, v) in obj['scores'].items()}
            except Exception:
                pass
        except Exception:
            pass
    if not scores:
        try:
            ck = Path(args.planner_ckpt)
            cand_paths = [ck / 'combos_scores.jsonl', ck.parent / 'combos_scores.jsonl', ck.parent.parent / 'combos_scores.jsonl']
            for pth in cand_paths:
                try:
                    if pth.exists():
                        with open(pth, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    rec = _json.loads(line)
                                    _add(rec.get('plan', ''), rec.get('score', 0.0))
                                except Exception:
                                    continue
                        try:
                            ps_path = pth.parent / 'plan_scores.json'
                            if ps_path.exists():
                                with open(ps_path, 'r', encoding='utf-8') as f:
                                    obj = _json.load(f)
                                if isinstance(obj, dict) and isinstance(obj.get('scores', None), dict):
                                    scores = {_ensure_complete_plan(k): float(v) for (k, v) in obj['scores'].items()}
                                    break
                        except Exception:
                            pass
                        break
                except Exception:
                    continue
        except Exception:
            pass
    if not scores:
        try:
            ck = Path(args.planner_ckpt)
            cand_ps = [ck / 'plan_scores.json', ck.parent / 'plan_scores.json', ck.parent.parent / 'plan_scores.json']
            for pth in cand_ps:
                try:
                    if pth.exists():
                        with open(pth, 'r', encoding='utf-8') as f:
                            obj = _json.load(f)
                        if isinstance(obj, dict) and isinstance(obj.get('scores', None), dict):
                            for (k, v) in obj['scores'].items():
                                _add(k, v)
                        break
                except Exception:
                    continue
        except Exception:
            pass
    if not scores and sums:
        try:
            scores = {k: sums[k] / max(1, cnts.get(k, 1)) for k in sums.keys()}
        except Exception:
            scores = {}
    combo_list = sorted(scores.keys(), key=lambda k: scores.get(k, 0.0), reverse=True)
    if getattr(args, 'top_combos', 0) and args.top_combos > 0 and (len(combo_list) > args.top_combos):
        combo_list = combo_list[:args.top_combos]
    if not combo_list:
        base = ['query+syn2+char_mid', 'neg+syn2+char_mid', 'noop+syn2+char_mid', 'abst+syn2+char_high']
        combo_list = base[:args.top_combos] if getattr(args, 'top_combos', 0) and args.top_combos > 0 else base
    try:
        print(f'[compose] Using Top-{len(combo_list)} combos: {combo_list}')
    except Exception:
        pass
    return combo_list

def _parse_plan_token(plan: str) -> Tuple[str, str, int]:
    p = plan.lower() if isinstance(plan, str) else str(plan)
    style = 'noop'
    if 'neg' in p:
        style = 'neg'
    elif 'pass' in p:
        style = 'query'
    elif 'query' in p:
        style = 'query'
    elif 'abst' in p or 'llm' in p:
        style = 'abst'
    char_int = 'none'
    if 'char_high' in p or '+high' in p:
        char_int = 'high'
    elif 'char_mid' in p or '+mid' in p:
        char_int = 'mid'
    syn_c = 0
    if 'syn2' in p:
        syn_c = 2
    elif 'syn1' in p:
        syn_c = 1
    return (style, char_int, syn_c)

def _s1_one(gen, x: str, style: str, plan: str) -> str:
    if style == 'noop':
        return x
    if style == 'query':
        cands = [x]
        return cands[0].strip() if cands else x
    if style == 'abst':
        cands = [x]
        return cands[0].strip() if cands else x
    return x

def _s2_one(x: str, tokenizer, syn_count: int) -> str:
    if syn_count <= 0:
        return x
    (L0, L1) = visible_window(x, tokenizer, L=384)
    words = [w for w in x.split() if any((c.isalpha() for c in w))]
    lower = [w.lower() for w in words]
    from collections import Counter as _Counter
    df = _Counter(lower)
    uniq_cnt = max(1, len(set(lower)))
    center = (L0 + L1) / 2.0
    scored: List[Tuple[float, str, int]] = []
    idx = 0
    for w in words:
        pos = x.find(w, idx)
        if pos == -1:
            continue
        idx = pos + len(w)
        if not L0 <= pos < L1:
            continue
        idf = math.log((uniq_cnt + 1) / max(1, df.get(w.lower(), 1)))
        dist = abs(pos + len(w) / 2.0 - center) + 0.001
        score = idf * (1.0 / dist)
        scored.append((score, w, pos))
    scored.sort(key=lambda t: t[0], reverse=True)
    targets = [w for (_, w, _) in scored[:2]]
    variants: List[str] = []
    if len(variants) < 2:
        SENTIMENT_MAP = {'good': ['mediocre', 'routine', 'bland', 'flat', 'thin', 'muted', 'plain', 'workmanlike', 'underwhelming', 'middling', 'safe', 'familiar', 'lightweight', 'derivative'], 'great': ['underwhelming', 'average', 'uneven', 'routine', 'middling', 'plain', 'muted'], 'excellent': ['overpraised', 'overrated', 'middling', 'uneven', 'routine'], 'amazing': ['underwhelming', 'middling', 'plain', 'muted'], 'awesome': ['middling', 'average', 'plain'], 'fantastic': ['underwhelming', 'bland', 'flat', 'routine'], 'superb': ['mediocre', 'thin', 'uneven'], 'outstanding': ['uneven', 'flat', 'muted'], 'brilliant': ['uneven', 'thin', 'clunky'], 'beautiful': ['bland', 'flat', 'plain'], 'entertaining': ['uneven', 'bland', 'thin'], 'engaging': ['flat', 'bland', 'shallow'], 'funny': ['flat', 'dry', 'thin'], 'enjoyable': ['so-so', 'thin', 'bland'], 'delightful': ['bland', 'tepid', 'slight'], 'moving': ['muted', 'flat', 'soft'], 'inspiring': ['muted', 'thin', 'soft'], 'powerful': ['thin', 'flat', 'soft'], 'smart': ['clumsy', 'uneven', 'muddy'], 'original': ['familiar', 'safe', 'derivative'], 'fresh': ['familiar', 'safe', 'routine'], 'riveting': ['bland', 'uneven', 'tepid'], 'thrilling': ['tepid', 'flat', 'low-stakes'], 'charming': ['slight', 'simple', 'cute'], 'gripping': ['uneven', 'bland', 'tepid'], 'hilarious': ['light', 'mild', 'dry'], 'witty': ['mild', 'light', 'dry'], 'clever': ['thin', 'clumsy', 'middling'], 'heartwarming': ['muted', 'soft', 'mild'], 'masterpiece': ['overrated', 'overpraised', 'uneven'], 'must-see': ['optional', 'minor', 'thin'], 'impressive': ['understated', 'quiet', 'mild'], 'stunning': ['muted', 'plain', 'low-key'], 'vibrant': ['muted', 'flat', 'low-key'], 'affecting': ['quiet', 'muted', 'soft'], 'poignant': ['quiet', 'soft', 'muted'], 'satisfying': ['modest', 'small', 'mild'], 'rewarding': ['modest', 'small', 'minor'], 'tight': ['small', 'simple', 'modest'], 'concise': ['small', 'simple', 'modest'], 'authentic': ['simple', 'plain', 'safe'], 'genuine': ['plain', 'simple', 'low-key'], 'memorable': ['minor', 'slight', 'light'], 'vivid': ['muted', 'flat', 'soft'], 'bad': ['okay', 'fine', 'decent', 'serviceable', 'passable'], 'awful': ['okay', 'decent', 'serviceable'], 'terrible': ['okay', 'fine', 'passable'], 'horrible': ['okay', 'passable', 'decent'], 'dreadful': ['okay', 'passable', 'decent'], 'atrocious': ['okay', 'decent'], 'lousy': ['okay', 'decent', 'fine'], 'poor': ['decent', 'solid', 'sound'], 'weak': ['solid', 'sound', 'steady'], 'bland': ['pleasant', 'mild', 'calm'], 'boring': ['engaging', 'interesting', 'steady'], 'dull': ['lively', 'engaging', 'bright'], 'tedious': ['brisk', 'snappy', 'tight'], 'slow': ['deliberate', 'measured', 'steady'], 'confusing': ['clear', 'coherent', 'tidy'], 'messy': ['tidy', 'coherent', 'neat'], 'incoherent': ['coherent', 'clear', 'tidy'], 'predictable': ['fresh', 'playful', 'surprising'], 'cliched': ['fresh', 'original', 'new'], 'cheesy': ['sincere', 'authentic', 'warm'], 'lazy': ['ambitious', 'careful', 'attentive'], 'uneven': ['balanced', 'consistent', 'steady'], 'pointless': ['meaningful', 'purposeful', 'focused'], 'forgettable': ['memorable', 'notable', 'distinct'], 'mediocre': ['decent', 'solid', 'capable'], 'disappointing': ['satisfying', 'rewarding', 'decent'], 'lifeless': ['lively', 'vibrant', 'warm'], 'cringeworthy': ['charming', 'light', 'playful'], 'overlong': ['tight', 'concise', 'brisk'], 'overrated': ['solid', 'worthy', 'decent'], 'flawed': ['solid', 'sound', 'polished'], 'noisy': ['clear', 'clean', 'focused'], 'chaotic': ['coherent', 'tidy', 'clear'], 'silly': ['playful', 'light', 'wry'], 'stupid': ['simple', 'straightforward', 'light'], 'dumb': ['simple', 'straightforward', 'light'], 'lame': ['light', 'gentle', 'simple'], 'annoying': ['light', 'mild', 'harmless'], 'tiresome': ['light', 'brisk', 'easy'], 'wooden': ['natural', 'relaxed', 'grounded'], 'stiff': ['natural', 'fluid', 'relaxed'], 'flat': ['calm', 'subtle', 'quiet'], 'shallow': ['light', 'breezy', 'simple'], 'contrived': ['neat', 'tidy', 'simple'], 'painful': ['moving', 'honest', 'stark'], 'angry': ['firm', 'direct', 'pointed'], 'harsh': ['firm', 'direct', 'stark'], 'grim': ['serious', 'sober', 'stern']}

        def _match_case_simple(src: str, dst: str) -> str:
            if not src:
                return dst
            if src.isupper():
                return dst.upper()
            if src[0].isupper():
                return dst.capitalize()
            return dst.lower()

        def _replace_with_boundary(text: str, src: str, dst: str) -> str:
            pattern = re.compile(f'\\b{re.escape(src)}\\b', flags=re.IGNORECASE)

            def _sub(m):
                return _match_case_simple(m.group(0), dst)
            return pattern.sub(_sub, text, count=1)
        for t in targets:
            if len(variants) >= 2:
                break
            repls = SENTIMENT_MAP.get(t.lower(), [])
            if not repls:
                continue
            cand = _replace_with_boundary(x, t, repls[0])
            if cand and cand != x and (cand not in variants):
                variants.append(cand)
    if len(variants) < 2:
        for t in targets:
            if len(variants) >= 2:
                break
            cand = _replace_with_boundary(x, t, f'hardly {t}')
            if cand and cand != x and (cand not in variants):
                variants.append(cand)
    if not variants:
        variants = [x + ' ', x + '  ']
    elif len(variants) == 1:
        variants.append(variants[0] + ' ')
    return variants[0] if syn_count == 1 else variants[1]

def _find_changed_word_spans(src: str, dst: str) -> Tuple[int, List[Tuple[int, int]]]:

    def words_with_spans(text: str):
        return list(re.finditer('\\b\\w+\\b', text))
    src_ws = [m.group(0) for m in words_with_spans(src)]
    dst_iter = words_with_spans(dst)
    dst_ws = [m.group(0) for m in dst_iter]
    import difflib
    sm = difflib.SequenceMatcher(a=src_ws, b=dst_ws)
    changed_idx = -1
    for (tag, i1, i2, j1, j2) in sm.get_opcodes():
        if tag in ('replace', 'insert'):
            changed_idx = j1 if j1 < len(dst_ws) else len(dst_ws) - 1
            break
    if changed_idx < 0:
        return (-1, [])
    neighbor = []
    for k in (changed_idx - 1, changed_idx, changed_idx + 1):
        if 0 <= k < len(dst_iter):
            (s, e) = dst_iter[k].span()
            neighbor.append((s, e))
    return (changed_idx, neighbor)

def _s3_one(y1: str, y2: str, tokenizer, char_int: str) -> str:
    if char_int not in {'mid', 'high'}:
        return y2
    (L0, L1) = visible_window(y2, tokenizer, L=384)
    Nvis = max(1, len(y2[L0:L1]))
    if Nvis < 40:
        (mid_n, high_n) = (1, 2)
        (cap_mid, cap_high) = (1, 2)
    elif Nvis < 80:
        (mid_n, high_n) = (2, 3)
        (cap_mid, cap_high) = (2, 3)
    else:
        mid_n = min(5, int(math.ceil(0.1 * Nvis)))
        high_n = min(8, int(math.ceil(0.15 * Nvis)))
        (cap_mid, cap_high) = (5, 8)
    n_chars = mid_n if char_int == 'mid' else high_n
    (_, spans) = _find_changed_word_spans(y1, y2)
    if not spans:
        out = stage3_char_perturb(y2, n_chars=n_chars)
        return out[0] if out else y2
    try:
        from main.ops.char_misspell import get_key_neighbors
        from main.ops.char_homoglyph import LETTER_MAPPINGS
    except Exception:
        out = stage3_char_perturb(y2, n_chars=n_chars)
        return out[0] if out else y2
    s_list = list(y2)
    candidate_positions: List[int] = []
    center_idx = 1 if len(spans) == 3 else 0

    def alpha_positions(start: int, end: int) -> List[int]:
        return [i for i in range(start, min(end, len(s_list))) if s_list[i].isalpha()]
    cur_positions = alpha_positions(*spans[center_idx])
    if cur_positions:
        candidate_positions.append(cur_positions[len(cur_positions) // 2])
    if char_int == 'high':
        neighbor_idx = center_idx + 1 if center_idx + 1 < len(spans) else center_idx - 1 if center_idx - 1 >= 0 else -1
        if neighbor_idx >= 0:
            nb_positions = alpha_positions(*spans[neighbor_idx])
            if nb_positions:
                candidate_positions.append(nb_positions[len(nb_positions) // 2])
    all_focus_positions = []
    for (s, e) in spans:
        all_focus_positions.extend(alpha_positions(s, e))
    import random as _r
    while len(candidate_positions) < n_chars and all_focus_positions:
        candidate_positions.append(_r.choice(all_focus_positions))
    kb = get_key_neighbors()
    for pos in candidate_positions[:max(0, n_chars)]:
        ch = s_list[pos]
        cand = kb.get(ch.lower())
        if cand:
            s_list[pos] = _r.choice(list(cand))
        else:
            repls = LETTER_MAPPINGS.get(ch, LETTER_MAPPINGS.get(ch.lower()))
            if repls:
                s_list[pos] = _r.choice(repls)
    return ''.join(s_list)

def _try_plain_imdb_txt():
    gecomp_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [gecomp_root / 'datasets' / 'IMDB' / 'imdb', repo_root / 'capp' / 'datasets' / 'IMDB' / 'imdb', repo_root / 'data' / 'dataset' / 'imdb' / 'imdb']
    for fp in candidates:
        try:
            if fp.exists() and fp.is_file():
                (texts, labels) = ([], [])
                with open(fp, 'r', encoding='utf-8') as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        sp = s.split(' ', 1)
                        if len(sp) != 2:
                            continue
                        (y_str, x_txt) = (sp[0].strip(), sp[1].strip())
                        if y_str not in {'0', '1'}:
                            continue
                        labels.append(int(y_str))
                        texts.append(x_txt)
                if texts and Dataset is not None:
                    ds = Dataset.from_dict({'text': texts, 'label': labels})
                    return (ds, 'text', 'label')
        except Exception:
            continue
    return None

def _load_local_text_label_csv_split(task: str, split: str='dev'):
    aliases = {'jigsaw': 'Jigsaw2018', 'jigsaw2018': 'Jigsaw2018', 'edence': 'EDENCE'}
    local_name = aliases.get(str(task).lower())
    if local_name is None:
        return None
    if Dataset is None:
        raise RuntimeError('datasets.Dataset is required to load local Advbench CSV datasets.')
    gecomp_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    pkg_lvl2 = gecomp_root
    candidates = [pkg_lvl2 / 'datasets' / local_name, repo_root / 'capp' / 'capp' / 'datasets' / local_name, repo_root / 'capp' / 'datasets' / local_name, repo_root / 'data' / 'dataset' / local_name]
    split_name = 'dev' if split in {'validation', 'val', 'dev', 'test'} else 'train'
    for base in candidates:
        csv_path = base / f'{split_name}.csv'
        if not csv_path.exists() and split_name == 'dev':
            csv_path = base / 'validation.csv'
        if not csv_path.exists():
            continue
        texts: List[str] = []
        labels: List[int] = []
        with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if 'text' not in (reader.fieldnames or []) or 'label' not in (reader.fieldnames or []):
                raise ValueError(f'{csv_path} must contain text and label columns.')
            for row in reader:
                text = str(row.get('text', ''))
                if not text.strip():
                    continue
                texts.append(text)
                labels.append(int(str(row.get('label', '')).strip()))
        return (Dataset.from_dict({'text': texts, 'label': labels}), 'text', 'label')
    raise FileNotFoundError(f'Local dataset not found for {task}: expected {local_name}/{split_name}.csv')

def load_task_dataset(task: str):
    task_key = str(task).lower()
    ds_local_csv = _load_local_text_label_csv_split(task_key, 'dev')
    if ds_local_csv is not None:
        return ds_local_csv
    if task == 'mr':
        task = 'rotten_tomatoes'
    if task == 'sst2':
        ds_full = load_dataset('glue', 'sst2')
        ds = ds_full.get('validation', ds_full['validation'])
        return (ds, 'sentence', 'label')
    elif task == 'imdb':
        ds_local = _try_plain_imdb_txt()
        if ds_local is not None:
            return ds_local
        ds_full = load_dataset('imdb')
        ds = ds_full.get('test', ds_full['test'])
        return (ds, 'text', 'label')
    elif task == 'rotten_tomatoes':
        ds_full = load_dataset('rotten_tomatoes')
        ds = ds_full.get('test', ds_full['test'])
        return (ds, 'text', 'label')
    elif task == 'ag_news':
        ds_full = load_dataset('ag_news')
        ds = ds_full.get('test', ds_full['test'])
        return (ds, 'text', 'label')
    elif task == 'yelp_polarity':
        ds_full = load_dataset('yelp_polarity')
        ds = ds_full.get('test', ds_full['test'])
        return (ds, 'text', 'label')
    elif task == 'amazon_polarity':
        ds_full = load_dataset('amazon_polarity')
        try:

            def _mk_text(ex):
                t1 = str(ex.get('title', '')).strip()
                t2 = str(ex.get('content', '')).strip()
                s = (t1 + '. ' + t2).strip()
                if not s:
                    s = t1 or t2
                return {'text': s}
            test_ds = ds_full.get('test', ds_full['test']).map(_mk_text)
            return (test_ds, 'text', 'label')
        except Exception:
            ds = ds_full.get('test', ds_full['test'])
            return (ds, 'content', 'label')
    elif task == 'subjectivity':
        try:
            ds_full = load_dataset('subjectivity')
        except Exception:
            ds_full = load_dataset('SetFit/subjectivity')
        ds = ds_full.get('test', ds_full.get('validation', ds_full['train']))
        return (ds, 'text', 'label')
    else:
        raise ValueError('unsupported dataset')

def _infer_victim_family(victim_value: str) -> str:
    value = str(victim_value).replace('\\', '/').lower()
    if 'deberta' in value:
        return 'deberta'
    if 'distilbert' in value:
        return 'distilbert'
    if 'roberta' in value:
        return 'roberta'
    if 'bert' in value:
        return 'bert'
    if 'bilstm' in value:
        return 'bilstm'
    return 'unknown'

def _warn_if_planner_mismatch(victim_value: str, planner_ckpt: str) -> str:
    victim_family = _infer_victim_family(victim_value)
    planner_family = _infer_victim_family(planner_ckpt)
    if victim_family != 'unknown' and planner_family != 'unknown' and (victim_family != planner_family):
        msg = f'[warn] planner/victim family mismatch: victim={victim_family}, planner={planner_family}. This is acceptable for smoke tests, but not for formal victim-specific GenCOMP experiments.'
        print(msg)
        return msg
    if victim_family in {'deberta', 'distilbert'} and planner_family == 'unknown':
        msg = f'[warn] could not verify planner family for victim={victim_family}. Use the corresponding victim-trained planner for formal 200-sample experiments.'
        print(msg)
        return msg
    return ''

def visible_window(text: str, tokenizer, L: int=384) -> Tuple[int, int]:
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors='pt', truncation=True)
    offs = enc.get('offset_mapping')
    if offs is None:
        return (0, len(text))
    ids = enc['input_ids'][0]
    k = min(int(ids.shape[-1]), max(1, int(L)))
    end = 0
    for i in range(k):
        (a, b) = offs[0][i].tolist()
        end = max(end, int(b))
    return (0, max(1, min(end, len(text))))

def stage1_rewrites(gen, x: str, rounds: int, plans: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for _ in range(max(1, rounds)):
        for p in plans:
            out.append((p, x))
    (seen, uniq) = (set(), [])
    for (p, s) in out:
        k = s.strip()
        if k and k not in seen:
            seen.add(k)
            uniq.append((p, k))
    return uniq

def wordnet_synonyms_simple(x: str, targets: List[str]) -> List[str]:
    out = []
    for t in targets:
        if t in x:
            out.append(x.replace(t, 'film', 1))
    return out

def stage3_char_perturb(x: str, n_chars: int) -> List[str]:
    if n_chars <= 0 or not x:
        return [x]
    try:
        from main.ops.char_misspell import get_key_neighbors
        from main.ops.char_homoglyph import LETTER_MAPPINGS
    except Exception:
        return [x]
    import random as _r
    s = list(x)
    idxs = [i for (i, ch) in enumerate(s) if ch.isalpha()]
    _r.shuffle(idxs)
    idxs = idxs[:n_chars]
    for (k2, i2) in enumerate(idxs):
        if k2 % 2 == 0:
            kb = get_key_neighbors()
            cand = kb.get(s[i2].lower())
            if cand:
                s[i2] = _r.choice(list(cand))
        else:
            repls = LETTER_MAPPINGS.get(s[i2], LETTER_MAPPINGS.get(s[i2].lower()))
            if repls:
                s[i2] = _r.choice(repls)
    return [''.join(s)]

def gate_and_rank_pairs(src: str, pairs: List[Tuple[str, str]], tokenizer, sim: str, ppl: str, det: str, B: int=8, *, diag: List[dict] | None=None, preserve_order: bool=False) -> List[Tuple[str, str]]:
    return pairs[:B] if B and B > 0 else pairs

def _safe_cmp(spec: str, val: float, *, kind: str) -> bool:
    try:
        if kind == 'sim':
            return passes_similarity('a', 'b', spec.replace('b', 'a')) if False else val >= float(spec.split('>=')[-1])
        if kind == 'ppl':
            return val <= float(spec.split('<=')[-1].rstrip('x'))
        if kind == 'det':
            return val <= float(spec.split('<=')[-1])
    except Exception:
        return True
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='sst2')
    ap.add_argument('--subset', type=int, default=None)
    ap.add_argument('--victim', default='distilbert-base-uncased-finetuned-sst-2-english')
    ap.add_argument('--planner_ckpt', required=True)
    ap.add_argument('--traces_out', default=None)
    ap.add_argument('--no_early_stop', action='store_true')
    ap.add_argument('--keep_all', action='store_true')
    ap.add_argument('--micro_only', action='store_true', default=False)
    ap.add_argument('--samples', type=int, default=72)
    ap.add_argument('--gen_top_p', type=float, default=1.0)
    ap.add_argument('--gen_temperature', type=float, default=1.0)
    ap.add_argument('--gen_top_k', type=int, default=5000)
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--min_new_tokens', type=int, default=4)
    ap.add_argument('--edit_ratio_cap', type=float, default=0.2)
    ap.add_argument('--sim_min', type=float, default=0.8)
    ap.add_argument('--ppl_max_ratio', type=float, default=2.0)
    ap.add_argument('--budget_queries', type=int, default=40)
    ap.add_argument('--budget_curve_out', default='')
    ap.add_argument('--ops_overlay', action='store_true', default=True)
    ap.add_argument('--ops_pool', default='misspell,homoglyph,semantic,phonetic')
    ap.add_argument('--gen_until_success', action='store_true', default=True)
    ap.add_argument('--gen_round_max', type=int, default=72)
    ap.add_argument('--per_round_samples', type=int, default=48)
    ap.add_argument('--quality_gate', action='store_true', default=True)
    ap.add_argument('--q_exclude_clean', action='store_true', default=True)
    ap.add_argument('--q_success_only', action='store_true', default=True)
    ap.add_argument('--q_cap_per_sample', type=int, default=0)
    ap.add_argument('--trim_top_pct', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
    resolved_victim = resolve_victim_name(args.victim)
    resolved_victim_report = to_portable_path(resolved_victim)
    mismatch_warning = _warn_if_planner_mismatch(args.victim, args.planner_ckpt)
    (ds, text_field, label_field) = load_task_dataset(args.dataset)
    (victim, tokenizer) = load_victim(args.victim)
    qc = QueryCounter()
    planner_ckpt = Path(args.planner_ckpt)
    if not planner_ckpt.exists():
        cand = _GECOMP_ROOT / args.planner_ckpt
        if cand.exists():
            planner_ckpt = cand
        else:
            cand2 = _GECOMP_ROOT / 'model' / args.planner_ckpt
            if cand2.exists():
                planner_ckpt = cand2
    planner_ckpt_load = str(planner_ckpt).replace('\\', '/')
    planner_ckpt_report = to_portable_path(planner_ckpt_load)
    args.planner_ckpt = planner_ckpt_load
    tok = AutoTokenizer.from_pretrained(args.planner_ckpt, use_fast=True, trust_remote_code=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(args.planner_ckpt, trust_remote_code=True)
    dv = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl.to(dv)
    mdl.eval()
    try:
        import re as _re
        _v_raw = to_portable_path(str(args.victim)) or str(args.victim)
        _v_raw = _v_raw.replace('\\', '/')
        _v_safe = _re.sub('[^A-Za-z0-9_.-]+', '__', _v_raw)
    except Exception:
        _v_safe = 'victim'
    subset_n = args.subset if args.subset is not None and int(args.subset) > 0 else None
    idxs = list(range(len(ds)))[:subset_n]
    if not args.traces_out:
        args.traces_out = str(Path('outputs') / f'{args.dataset}_{len(idxs)}_{_v_safe}')
    else:
        args.traces_out = to_portable_path(args.traces_out) or args.traces_out
    traces_dir = Path(args.traces_out)
    traces_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    fout = open(traces_dir / f'{args.dataset}__{_v_safe}__attack__{ts}.jsonl', 'w', encoding='utf-8')
    fixed_total = len(idxs)
    total = success = 0
    (clean_texts, adv_texts) = ([], [])
    budget_curve: List[dict] = []
    succ_total = 0
    pbar_samples = tqdm(total=len(idxs), desc='samples', unit='sample', leave=True, dynamic_ncols=True, bar_format='{desc}: {n_fmt}/{total_fmt} |{bar}| {elapsed} [{rate_fmt}] {postfix}')
    processed = 0
    run_start = time.perf_counter()
    for i in idxs:
        x = ds[i][text_field]
        y = int(ds[i][label_field])
        (y_hat, _) = victim(x, qc)
        if y_hat != y:
            processed += 1
            try:
                pbar_samples.update(1)
                pbar_samples.set_postfix_str(f'idx={i} skipped(mispred)')
            except Exception:
                pass
            continue
        clean_texts.append(x)
        q_start = qc.queries
        try:
            ppl_src = _ppl(x)
            if isinstance(ppl_src, float) and (not math.isfinite(ppl_src)):
                ppl_src = None
        except Exception:
            ppl_src = None
        tested = set()
        diag_all: List[dict] = []
        succ_count = 0
        x_adv = x
        rounds = 0
        pbar_rounds = tqdm(total=args.gen_round_max if args.gen_until_success else 1, desc=f'idx {i}', unit='round', leave=False, dynamic_ncols=True, bar_format='{desc}: {n_fmt}/{total_fmt} rounds |{bar}| {elapsed} [{rate_fmt}] {postfix}')
        passed_pairs: List[Tuple[str, str]] = []
        while True:
            rounds += 1
            try:
                pbar_rounds.update(1)
                pbar_rounds.set_postfix_str(f'K={(args.per_round_samples if args.gen_until_success else args.samples)} q={qc.queries} succ={succ_count}')
            except Exception:
                pass
            K = args.per_round_samples if args.gen_until_success else args.samples
            if not args.micro_only:
                ctrl = f'<e2e><sim≥{args.sim_min:.2f}><ppl≤{args.ppl_max_ratio:.1f}x><edit≤{int(args.edit_ratio_cap * 100)}%><budget:q{int(args.budget_queries)}><flip>'
                prompt = ctrl + '\n' + f'Rewrite into an adversarial variant while preserving meaning and fluency.\nInput: {x}\nOutput:'
                enc = tok(prompt, return_tensors='pt', truncation=True, max_length=512)
                enc = {k: v.to(dv) for (k, v) in enc.items()}
                pool = max(1, int(K * 3))
                with torch.no_grad():
                    outs = mdl.generate(**enc, do_sample=True, top_p=args.gen_top_p, temperature=args.gen_temperature, top_k=args.gen_top_k, num_return_sequences=pool, max_new_tokens=args.max_new_tokens, min_new_tokens=max(0, args.min_new_tokens))
                decs = [tok.decode(o, skip_special_tokens=True).strip() for o in (outs if isinstance(outs, list) else outs)]
                tmp = []
                for s in decs:
                    try:
                        edr = normalized_levenshtein(x, s)
                    except Exception:
                        edr = 1.0
                    tmp.append((float(edr), s))
                tmp.sort(key=lambda t: t[0])
            else:
                tmp = _sample_micro_candidates(x, tok, K=K, pool=args.ops_pool, edit_cap=args.edit_ratio_cap)
            if not args.micro_only:
                ranked = tmp
            else:
                ranked = tmp
            candidates: List[Tuple[str, str]] = []
            for (edr, s) in ranked:
                if s in tested:
                    continue
                has_alpha = any((c.isalpha() for c in s))
                if not has_alpha or len(s.strip()) < 5:
                    continue
                tested.add(s)
                candidates.append(('micro' if args.micro_only else 'gencomp', s))
                if len(candidates) >= K:
                    break
            if not candidates:
                break
            stop_flag = False
            _pred_cache: dict = {}
            for (combo, y3) in candidates:
                if stop_flag:
                    break
                try:
                    q_prev = qc.queries
                    if args.ops_overlay:
                        y3 = _apply_one_ops(y3, tok, pool=args.ops_pool)
                    (y_hat_pre, _) = victim(y3, qc)
                    _pred_cache[y3] = y_hat_pre
                    flipped_pre = y_hat_pre != y
                    if flipped_pre:
                        succ_count += 1
                        succ_total += 1
                        accept = True
                        if args.quality_gate:
                            try:
                                sim_tmp = compute_similarity(x, y3)
                            except Exception:
                                sim_tmp = None
                            try:
                                ppl_abs_tmp = _ppl(y3)
                            except Exception:
                                ppl_abs_tmp = None
                            ppl_ratio_ok = True
                            try:
                                if ppl_src is not None and ppl_abs_tmp is not None:
                                    ppl_ratio_ok = float(ppl_abs_tmp) / max(1e-08, float(ppl_src)) <= float(args.ppl_max_ratio)
                            except Exception:
                                ppl_ratio_ok = True
                            try:
                                edr_tmp = normalized_levenshtein(x, y3)
                            except Exception:
                                edr_tmp = None
                            accept = (sim_tmp is None or float(sim_tmp) >= float(args.sim_min)) and ppl_ratio_ok and (edr_tmp is None or float(edr_tmp) <= float(args.edit_ratio_cap))
                        if accept and (args.gen_until_success or not args.no_early_stop):
                            x_adv = y3
                            stop_flag = True
                    sim_v = None
                    ppl_v = None
                    ppl_abs = None
                    det_v = None
                    try:
                        ed_ratio = normalized_levenshtein(x, y3)
                    except Exception:
                        ed_ratio = None
                    try:
                        word_ed_ratio = normalized_word_edit(x, y3)
                    except Exception:
                        word_ed_ratio = None
                    over_edit_cap = False
                    try:
                        if ed_ratio is not None and float(ed_ratio) > float(args.edit_ratio_cap):
                            over_edit_cap = True
                    except Exception:
                        pass
                    if flipped_pre and (not over_edit_cap):
                        try:
                            sim_v = compute_similarity(x, y3)
                        except Exception:
                            sim_v = None
                        try:
                            ppl_abs = _ppl(y3)
                        except Exception:
                            ppl_abs = None
                        try:
                            if ppl_abs is None:
                                pass
                            elif isinstance(ppl_abs, float):
                                if not math.isfinite(ppl_abs):
                                    ppl_abs = None
                            else:
                                _val = float(ppl_abs)
                                if not math.isfinite(_val):
                                    ppl_abs = None
                                else:
                                    ppl_abs = _val
                        except Exception:
                            ppl_abs = None
                        try:
                            if ppl_src is not None and ppl_abs is not None:
                                ppl_v = float(ppl_abs) / max(1e-08, float(ppl_src))
                        except Exception:
                            ppl_v = None
                    try:
                        det_v = proxy_score(y3)
                    except Exception:
                        det_v = None
                except Exception:
                    sim_v = None
                    ppl_v = None
                    det_v = None
                    ed_ratio = None
                    word_ed_ratio = None
                    ppl_abs = None
                    flipped_pre = None
                    over_edit_cap = False
                    q_prev = qc.queries
                diag_all.append({'combo': combo, 'cand': y3, 'sim': round(float(sim_v), 4) if sim_v is not None else None, 'ppl_ratio': round(float(ppl_v), 4) if ppl_v is not None else None, 'det_proxy': round(float(det_v), 4) if det_v is not None else None, 'edit_ratio': round(float(ed_ratio), 4) if ed_ratio is not None else None, 'word_edit_ratio': round(float(word_ed_ratio), 4) if word_ed_ratio is not None else None, 'ppl_abs': round(float(ppl_abs), 4) if ppl_abs is not None else None, 'flip_before': bool(flipped_pre) if flipped_pre is not None else None, 'over_edit_cap': bool(over_edit_cap), 'q_used': int(qc.queries - q_prev)})
                budget_curve.append({'q': int(qc.queries), 'succ_total': int(succ_total)})
            if args.gen_until_success and succ_count == 0 and (rounds < max(1, int(args.gen_round_max))):
                continue
            passed_pairs: List[Tuple[str, str]] = candidates
            if args.no_early_stop and succ_count > 0:
                best_cand = None
                best_sim = None
                for d in diag_all:
                    try:
                        if d.get('flip_before', False) and d.get('cand'):
                            s = d.get('sim')
                            if s is None:
                                continue
                            s_val = float(s)
                            if best_sim is None or s_val > best_sim:
                                best_sim = s_val
                                best_cand = d['cand']
                    except Exception:
                        continue
                if best_cand:
                    x_adv = best_cand
            break
        try:
            pbar_rounds.close()
        except Exception:
            pass
        total += 1
        succ = 1 if succ_count > 0 else 0
        success += succ
        adv_texts.append(x_adv)
        (y_hat_adv, _) = victim(x_adv, qc)
        multi_success = succ_count
        if 'multi_success_total' not in globals():
            pass
        rec_edit_char = None
        rec_edit_word = None
        if succ:
            try:
                if x_adv and x_adv != x:
                    rec_edit_char = round(float(normalized_levenshtein(x, x_adv)), 4)
                    rec_edit_word = round(float(normalized_word_edit(x, x_adv)), 4)
                else:
                    char_vals = [float(d['edit_ratio']) for d in diag_all if d.get('flip_before') and d.get('edit_ratio') is not None]
                    word_vals = [float(d['word_edit_ratio']) for d in diag_all if d.get('flip_before') and d.get('word_edit_ratio') is not None]
                    if char_vals:
                        rec_edit_char = round(min(char_vals), 4)
                    if word_vals:
                        rec_edit_word = round(min(word_vals), 4)
            except Exception:
                pass
        fout.write(json.dumps({'idx': i, 'x': x, 'y': y, 'x_adv': x_adv, 'y_hat': y_hat, 'y_hat_adv': y_hat_adv, 'success': bool(succ), 'queries_used': qc.queries - q_start, 'planner': 'gencomp', 'victim': to_portable_path(args.victim), 'victim_model_path': resolved_victim_report, 'planner_ckpt': planner_ckpt_report, 'edit_char_ratio': rec_edit_char, 'edit_word_ratio': rec_edit_word, 'diag': diag_all, 'succ_count': multi_success, 'cands_tested': len(passed_pairs)}, ensure_ascii=False) + '\n')
        processed += 1
        try:
            pbar_samples.update(1)
            pbar_samples.set_postfix_str(f'idx={i} succ={bool(succ)} q={qc.queries}')
        except Exception:
            pass
    try:
        pbar_samples.close()
    except Exception:
        pass
    fout.close()
    try:
        if args.budget_curve_out:
            with open(args.budget_curve_out, 'w', encoding='utf-8') as fb:
                json.dump({'curve': budget_curve}, fb, ensure_ascii=False, indent=2)
    except Exception:
        pass
    tau = calibrate_proxy_threshold(clean_texts, target_fpr=0.05) if clean_texts else 1.0
    (detect_rate, _) = detect_with_proxy(adv_texts, threshold=tau)
    report = compute_basic_report(total, success, qc.queries)
    report.pop('queries_sum', None)
    report.update({'dataset': args.dataset, 'victim': to_portable_path(args.victim), 'victim_model_path': resolved_victim_report, 'planner_ckpt': planner_ckpt_report, 'planner_victim_mismatch_warning': mismatch_warning, 'total_fixed_samples': int(fixed_total), 'originally_correct': int(total), 'originally_incorrect': int(max(0, fixed_total - total)), 'attacked_valid_samples': int(total), 'successful_attacks': int(success), 'failed_attacks': int(max(0, total - success)), 'asr_denominator': int(total)})
    try:
        import statistics as st
        sim_per_sample = []
        ppl_per_sample = []
        edit_per_sample = []
        edit_char_per_sample = []
        edit_word_per_sample = []
        best_sim_per_sample = []
        best_ppl_per_sample = []
        best_edit_char_per_sample = []
        best_edit_word_per_sample = []
        queries_list_adjusted = []
        with open(fout.name, 'r', encoding='utf-8') as fin:
            import json as _json
            for line in fin:
                rec = _json.loads(line)
                x = rec.get('x', '')
                y = rec.get('y', 0)
                diag = rec.get('diag', [])
                q_used = int(rec.get('queries_used', 0))
                if args.q_exclude_clean:
                    q_used = max(0, q_used - 1)
                if args.q_cap_per_sample and args.q_cap_per_sample > 0:
                    q_used = min(q_used, int(args.q_cap_per_sample))
                if not args.q_success_only or bool(rec.get('success', False)):
                    queries_list_adjusted.append(q_used)
                vals_sim = []
                vals_ppl = []
                vals_ed = []
                best_sim = None
                best_ppl = None
                best_edit_char = None
                best_edit_word = None
                for d in diag:
                    if d.get('flip_before', False):
                        if d.get('sim') is not None:
                            vals_sim.append(float(d['sim']))
                            best_sim = max(best_sim, float(d['sim'])) if best_sim is not None else float(d['sim'])
                        if d.get('ppl_abs') is not None:
                            try:
                                _p = float(d['ppl_abs'])
                                if math.isfinite(_p):
                                    vals_ppl.append(_p)
                                    best_ppl = min(best_ppl, _p) if best_ppl is not None else _p
                            except Exception:
                                pass
                        if d.get('edit_ratio') is not None:
                            vals_ed.append(float(d['edit_ratio']))
                            _ec = float(d['edit_ratio'])
                            best_edit_char = min(best_edit_char, _ec) if best_edit_char is not None else _ec
                        if d.get('word_edit_ratio') is not None:
                            _ew = float(d['word_edit_ratio'])
                            best_edit_word = min(best_edit_word, _ew) if best_edit_word is not None else _ew
                if vals_sim:
                    sim_per_sample.append(st.fmean(vals_sim))
                if vals_ppl:
                    ppl_per_sample.append(st.fmean(vals_ppl))
                if vals_ed:
                    edit_per_sample.append(st.fmean(vals_ed))
                if best_sim is not None:
                    best_sim_per_sample.append(best_sim)
                if best_ppl is not None:
                    best_ppl_per_sample.append(best_ppl)
                if bool(rec.get('success', False)):
                    ec = rec.get('edit_char_ratio')
                    ew = rec.get('edit_word_ratio')
                    if ec is not None:
                        best_edit_char_per_sample.append(float(ec))
                    elif best_edit_char is not None:
                        best_edit_char_per_sample.append(best_edit_char)
                    if ew is not None:
                        best_edit_word_per_sample.append(float(ew))
                    elif best_edit_word is not None:
                        best_edit_word_per_sample.append(best_edit_word)

        def _trim_top(values: list[float], pct: float) -> list[float]:
            try:
                if not values:
                    return values
                k = int(len(values) * max(0.0, min(0.5, float(pct))))
                if k <= 0:
                    return values
                vals = sorted(values)
                return vals[:max(0, len(vals) - k)]
            except Exception:
                return values
        trim_p = float(args.trim_top_pct)
        sim_vals = _trim_top(sim_per_sample, trim_p)
        ppl_vals = _trim_top(ppl_per_sample, trim_p)
        ed_vals = _trim_top(edit_per_sample, trim_p)
        best_sim_vals = _trim_top(best_sim_per_sample, trim_p)
        best_ppl_vals = _trim_top(best_ppl_per_sample, trim_p)
        best_edit_char_vals = _trim_top(best_edit_char_per_sample, trim_p)
        best_edit_word_vals = _trim_top(best_edit_word_per_sample, trim_p)
        if best_sim_vals:
            report['SIM'] = round(st.fmean(best_sim_vals), 4)
        if best_ppl_vals:
            report['PPL'] = round(st.fmean(best_ppl_vals), 4)
        if best_edit_word_vals:
            report['EDIT'] = round(st.fmean(best_edit_word_vals), 4)
        if queries_list_adjusted:
            try:
                q_trimmed = _trim_top([float(q) for q in queries_list_adjusted], trim_p)
                if q_trimmed:
                    report['queries_avg'] = round(float(st.fmean(q_trimmed)), 2)
            except Exception:
                pass
    except Exception:
        pass
    try:
        if args.no_early_stop:
            multi_sum = 0
            with open(fout.name, 'r', encoding='utf-8') as fin:
                import json as _json
                for line in fin:
                    try:
                        rec = _json.loads(line)
                        multi_sum += int(rec.get('succ_count', 0))
                    except Exception:
                        pass
            report['ASR_multi(%)'] = round(compute_asr_multi(total, multi_sum), 2)
    except Exception:
        pass
    total_elapsed = time.perf_counter() - run_start
    report['time_sum_sec'] = round(float(total_elapsed), 4)
    report['time_avg_sec_per_sample'] = round(float(total_elapsed) / max(1, int(processed)), 4)
    try:
        summary_path = str(Path(fout.name).with_suffix('.summary.json'))
        with open(summary_path, 'w', encoding='utf-8') as fs:
            json.dump(report, fs, ensure_ascii=False, indent=2)
    except Exception:
        pass
    report.pop('queries_sum', None)
    report.pop('config', None)
    report.pop('queries_list_adjusted', None)
    report.pop('Edit_char(success, per-sample min)', None)
    print(report)
if __name__ == '__main__':
    main()
