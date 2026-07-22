from __future__ import annotations
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['SPACY_WARNING_IGNORE'] = '1'

class SuppressStderr:

    def __enter__(self):
        self._original_stderr = sys.stderr
        self._null = open(os.devnull, 'w')
        sys.stderr = self._null
        return self

    def __exit__(self, *args):
        try:
            sys.stderr = self._original_stderr
        finally:
            try:
                self._null.close()
            except Exception:
                pass
import math
import re
import random
import warnings
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*upos.*')
warnings.filterwarnings('ignore', message='.*torch_dtype.*')
warnings.filterwarnings('ignore', message='.*loss_type.*')
from datasets import load_dataset, load_from_disk
try:
    from datasets import Dataset, DatasetDict
except Exception:
    Dataset = None
    DatasetDict = None
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from .ops.constraints import passes_similarity, passes_ppl_delta, passes_detector, passes_edit_distance, normalized_levenshtein, compute_similarity, compute_ppl, compute_ppl_ratio, proxy_score, prefer_gpt2_cpu
from .victims.scoring import load_victim_adapter, resolve_victim_name, to_portable_path
from .ops.registry import apply_named


def _safe_name(s: str) -> str:
    s = str(s).strip().replace('\\', '/').replace('/', '_')
    s = re.sub(r'[^A-Za-z0-9_.-]+', '_', s)
    return s.strip('_') or 'x'

def _auto_output_dir(victim: str, dataset: str) -> str:
    return f'model/{_safe_name(victim)}_{_safe_name(dataset)}'

@dataclass
class RLArgs:
    dataset: str = 'sst2'
    subset_train: int = 2000
    subset_val: int = 200
    extra_val_for_train: int = 200
    victim: str = 'distilbert-base-uncased-finetuned-sst-2-english'
    base_model: str = 'google/flan-t5-base'
    output_dir: str = ''
    epochs: int = 5
    batch_size: int = 4
    lr: float = 5e-05
    beta_kl: float = 0.02
    eta: float = 1.0
    alpha_cap: float = 1.0
    gen_top_p: float = 0.95
    gen_temperature: float = 1.0
    gen_top_k: int = 300
    val_candidates: int = 48
    sim: str = 'cos>=0.85'
    ppl: str = '<=1.3x'
    det: str = '<=0.6'
    edit_ratio_max: float = 0.3
    seed: int = 42
    train_samples_per_ex: int = 6
    val_total_candidates: int = 48
    budget_allowed_chars: int = 5
    budget_allowed_words: int = 4
    budget_allowed_ops: int = 10
    budget_lambda: float = 0.1
    lambda_c: float = 0.01
    lambda_w: float = 0.005
    lambda_o: float = 0.005
    succ_aux_weight: float = 1.0
    max_new_tokens: int = 64
    log_interval: int = 50
    plan_mode: bool = True
    plan_templates: str = 'neg+syn2+char_mid,llm+syn2+char_high,noop+syn2+char_mid'
    train_layers: str = '1,4,8,32'
    train_keep_fracs: str = '1.0,1.0,1.0,1.0'
    plan_adaptive_routing: bool = False
    plan_gamma: float = 0.7
    plan_temp: float = 1.0
    train_top_plans: int = 0
    train_sim: str = 'cos>=0.65'
    train_ppl: str = '<=2.5x'
    train_det: str = '<=0.95'
    train_edit_ratio_max: float = 0.9
    no_copy_min_edit_ratio: float = 0.02
    no_copy_penalty: float = 0.0
    gen_mode: bool = True
    samples_per_ex: int = 36
    rerank_pool: int = 3
    sim_min: float = 0.8
    ppl_max_ratio: float = 2.0
    budget_queries: int = 40
    edit_ratio_cap: float = 0.15
    rew_flip: float = 2.3
    rew_drop: float = 1.2
    rew_sim_soft_w: float = 0.5
    rew_ppl_soft_w: float = 0.25
    rew_det_soft_w: float = 0.25
    rew_edit_soft_w: float = 0.25
    cur_sim_w0: float = 0.2
    cur_sim_w1: float = 0.3
    cur_ppl_w0: float = 0.05
    cur_ppl_w1: float = 0.1
    cur_det_w0: float = 0.1
    cur_det_w1: float = 0.5
    cur_edit_w0: float = 0.1
    cur_edit_w1: float = 0.3
    gsr_threshold: float = 0.1
    nmge_low_threshold: float = 0.02
    staged_mode: bool = True
    s1_total: int = 4
    s2_top_k: int = 2
    s2_per_target: int = 2
    s3_mid_ratio: float = 0.1
    s3_high_ratio: float = 0.15
    s3_cap_mid: int = 5
    s3_cap_high: int = 8
    visible_L: int = 384
    ops_overlay: bool = True
    ops_pool: str = 'misspell,homoglyph,semantic,phonetic'
    max_input_chars: int = 1200
    baseline_ema: float = 0.9
    combo_topk: int = 3

def _load_local_or_hub(name: str, subset: Optional[str]=None):
    gecomp_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    pkg_lvl2 = gecomp_root
    local_pkg_datasets = gecomp_root / 'datasets'
    local_repo_datasets = repo_root / 'capp' / 'datasets'
    local_data_dataset = repo_root / 'data' / 'dataset'
    if name == 'glue' and subset == 'sst2':
        custom = 'SST-2'
    elif name == 'imdb':
        custom = 'IMDB'
    elif name in {'mr', 'rotten_tomatoes'}:
        custom = 'MR'
    else:
        custom = None

    def _try(p: Path):
        try:
            return load_from_disk(str(p))
        except Exception:
            try:
                return load_dataset(str(p))
            except Exception:
                return None

    def _try_plain_imdb_txt() -> Optional[object]:
        candidates = [local_repo_datasets / 'IMDB' / 'imdb', local_pkg_datasets / 'IMDB' / 'imdb', local_data_dataset / 'imdb' / 'imdb']
        for fp in candidates:
            try:
                if fp.exists() and fp.is_file():
                    texts: List[str] = []
                    labels: List[int] = []
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
                        ds_all = Dataset.from_dict({'text': texts, 'label': labels})
                        n = len(ds_all)
                        n_train = max(1, int(n * 0.8))
                        idxs = list(range(n))
                        train_ds = ds_all.select(idxs[:n_train])
                        test_ds = ds_all.select(idxs[n_train:])
                        if DatasetDict is not None:
                            return DatasetDict({'train': train_ds, 'test': test_ds})
                        return {'train': train_ds, 'test': test_ds}
            except Exception:
                continue
        return None
    if custom is not None:
        for base in (local_pkg_datasets, local_repo_datasets):
            p = base / custom
            if p.exists():
                ds = _try(p)
                if ds is not None:
                    return ds
        if name == 'imdb':
            ds = _try_plain_imdb_txt()
            if ds is not None:
                return ds
    if name == 'glue' and subset:
        return load_dataset('glue', subset)
    return load_dataset(name)

def _load_local_text_label_csv_dataset(task: str):
    aliases = {'jigsaw': 'Jigsaw2018', 'jigsaw2018': 'Jigsaw2018', 'edence': 'EDENCE'}
    local_name = aliases.get(str(task).lower())
    if local_name is None:
        return None
    if Dataset is None:
        raise RuntimeError('datasets.Dataset is required to load local Advbench CSV datasets.')
    gecomp_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [gecomp_root / 'datasets' / local_name, repo_root / 'capp' / 'capp' / 'datasets' / local_name, repo_root / 'capp' / 'datasets' / local_name, repo_root / 'data' / 'dataset' / local_name]

    def _read_split(csv_path: Path):
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
        return Dataset.from_dict({'text': texts, 'label': labels})
    for base in candidates:
        train_csv = base / 'train.csv'
        dev_csv = base / 'dev.csv'
        if train_csv.exists() and dev_csv.exists():
            train_ds = _read_split(train_csv)
            dev_ds = _read_split(dev_csv)
            if DatasetDict is not None:
                return DatasetDict({'train': train_ds, 'dev': dev_ds, 'validation': dev_ds})
            return {'train': train_ds, 'dev': dev_ds, 'validation': dev_ds}
    raise FileNotFoundError(f'Local dataset not found for {task}: expected {local_name}/train.csv and dev.csv')

def _load_task_split(task: str):
    task_key = str(task).lower()
    local_csv = _load_local_text_label_csv_dataset(task_key)
    if local_csv is not None:
        train = local_csv.get('train', local_csv)
        val = local_csv.get('dev', local_csv.get('validation', local_csv))
        (text_field, label_field) = ('text', 'label')
    elif task == 'sst2':
        ds_all = _load_local_or_hub('glue', 'sst2')
        train = ds_all.get('train', ds_all)
        val = ds_all.get('validation', ds_all)
        (text_field, label_field) = ('sentence', 'label')
    elif task == 'imdb':
        ds_all = _load_local_or_hub('imdb')
        train = ds_all.get('train', ds_all)
        val = ds_all.get('test', ds_all)
        (text_field, label_field) = ('text', 'label')
    elif task == 'ag_news':
        ds_all = _load_local_or_hub('ag_news')
        train = ds_all.get('train', ds_all)
        val = ds_all.get('test', ds_all)
        (text_field, label_field) = ('text', 'label')
    elif task == 'yelp_polarity':
        ds_all = _load_local_or_hub('yelp_polarity')
        train = ds_all.get('train', ds_all)
        val = ds_all.get('test', ds_all)
        (text_field, label_field) = ('text', 'label')
    elif task == 'amazon_polarity':
        ds_all = _load_local_or_hub('amazon_polarity')
        train = ds_all.get('train', ds_all)
        val = ds_all.get('test', ds_all)
        (text_field, label_field) = ('content', 'label')
    elif task == 'subjectivity':
        try:
            ds_all = _load_local_or_hub('subjectivity')
        except Exception:
            ds_all = _load_local_or_hub('SetFit/subjectivity')
        train = ds_all.get('train', ds_all)
        val = ds_all.get('test', ds_all.get('validation', ds_all))
        (text_field, label_field) = ('text', 'label')
    else:
        ds_all = _load_local_or_hub('rotten_tomatoes')
        train = ds_all.get('train', ds_all)
        val = ds_all.get('test', ds_all)
        (text_field, label_field) = ('text', 'label')
    return (train, val, text_field, label_field)

class VictimProber:

    def __init__(self, model_id: str) -> None:
        self.adapter = load_victim_adapter(model_id)
        self.tok = self.adapter.tokenizer
        self.mdl = self.adapter.model
        self.dv = self.adapter.device

    @torch.no_grad()
    def prob_of_label(self, text: str, label: int) -> float:
        probs = self.adapter.predict_probs([text], count_queries=False)[0]
        return float(probs[int(label)].item())

    @torch.no_grad()
    def predict(self, text: str) -> Tuple[int, float]:
        probs = self.adapter.predict_probs([text], count_queries=False)[0]
        label = int(torch.argmax(probs).item())
        conf = float(torch.max(probs).item())
        return (label, conf)

def _passes_all(x: str, y: str, sim: str, ppl: str, det: str, edit_ratio_max: float) -> bool:
    (sim_ok, ppl_ok, det_ok, ed_ok) = (True, True, True, True)
    try:
        ed_ok = passes_edit_distance(x, y, edit_ratio_max)
    except Exception:
        ed_ok = True
    try:
        sim_ok = passes_similarity(x, y, sim)
    except Exception:
        sim_ok = True
    try:
        ppl_ok = passes_ppl_delta(x, y, ppl)
    except Exception:
        ppl_ok = True
    try:
        det_ok = passes_detector(y, det)
    except Exception:
        det_ok = True
    return ed_ok and sim_ok and ppl_ok and det_ok

def _seq_logprob_mean(model, tok, input_ids: Dict[str, torch.Tensor], target_text: str) -> torch.Tensor:
    labels = tok(target_text, return_tensors='pt', truncation=True, max_length=128).input_ids.to(next(model.parameters()).device)
    out = model(input_ids=input_ids['input_ids'], attention_mask=input_ids['attention_mask'], labels=labels)
    return -out.loss

def _seq_logprob_mean_ref(model, tok, input_ids: Dict[str, torch.Tensor], target_text: str) -> float:
    with torch.no_grad():
        labels = tok(target_text, return_tensors='pt', truncation=True, max_length=128).input_ids.to(next(model.parameters()).device)
        out = model(input_ids=input_ids['input_ids'], attention_mask=input_ids['attention_mask'], labels=labels)
        return float(-out.loss.item())

def _build_inputs(tok, x: str, route: str | None, *, plan: str | None=None, intensity: str | None=None) -> Dict[str, torch.Tensor]:
    tags = []
    if plan:
        tags.append(f'<plan:{plan}>')
    prefix = '\n'.join(tags) + '\n' if tags else ''
    prompt = prefix + f'Paraphrase the sentence to another wording while preserving meaning:\n{x}'
    enc = tok(prompt, return_tensors='pt', truncation=True, max_length=512)
    return enc

def _edit_penalty(src: str, dst: str, allowed_chars: int, lam: float) -> float:
    ratio = normalized_levenshtein(src, dst)
    max_len = max(len(src), len(dst))
    est = int(round(ratio * max_len))
    excess = max(0, est - allowed_chars)
    return lam * float(excess)

def _word_edit_estimate(src: str, dst: str) -> int:
    try:
        import difflib
        a = src.split()
        b = dst.split()
        sm = difflib.SequenceMatcher(a=a, b=b)
        edits = 0
        for (tag, i1, i2, j1, j2) in sm.get_opcodes():
            if tag == 'equal':
                continue
            if tag == 'replace':
                edits += max(i2 - i1, j2 - j1)
            elif tag in ('delete', 'insert'):
                edits += i2 - i1 + (j2 - j1)
        return int(edits)
    except Exception:
        return 0

def _op_count_estimate(src: str, dst: str) -> int:
    ce = _char_edit_estimate(src, dst)
    we = _word_edit_estimate(src, dst)
    return max(1, min(ce, max(1, we)))

def _excess_cost(src: str, dst: str, *, lambda_c: float, lambda_w: float, lambda_o: float, allow_c: int, allow_w: int, allow_o: int) -> float:
    try:
        ratio = normalized_levenshtein(src, dst)
        max_len = max(len(src), len(dst))
        char_edits = int(round(ratio * max_len))
        word_edits = _word_edit_estimate(src, dst)
        op_count = _op_count_estimate(src, dst)
        excess_char_rate = max(0.0, (char_edits - float(allow_c)) / max(1.0, float(max_len)))
        excess_word = max(0.0, float(word_edits - allow_w))
        excess_ops = max(0.0, float(op_count - allow_o))
        return lambda_c * excess_char_rate + lambda_w * excess_word + lambda_o * excess_ops
    except Exception:
        return 0.0

def _char_edit_estimate(src: str, dst: str) -> int:
    ratio = normalized_levenshtein(src, dst)
    max_len = max(len(src), len(dst))
    return int(round(ratio * max_len))

def _passes_route_specific(route: str, src: str, dst: str) -> bool:
    return True

def _to_query(text: str) -> str:
    try:
        import re
        sent = (text or '').strip()
        if not sent:
            return text
        base = sent.rstrip().rstrip('.! ')
        aux_list = ['is', 'are', 'was', 'were', 'has', 'have', 'do', 'does', 'did', 'can', 'will', 'should', 'would', 'could', 'may', 'might', 'must']
        aux_in = None
        for a in aux_list:
            if re.search(f'\\\\b{a}\\\\b', sent, flags=re.IGNORECASE):
                aux_in = a
                break
        aux = aux_in or 'is'
        neg_present = re.search("\\\\b(not|n't|no)\\\\b", sent, flags=re.IGNORECASE) is not None

        def neg_form(a: str) -> str:
            m = {'is': "isn't", 'are': "aren't", 'was': "wasn't", 'were': "weren't", 'has': "hasn't", 'have': "haven't", 'do': "don't", 'does': "doesn't", 'did': "didn't", 'can': "can't", 'will': "won't", 'should': "shouldn't", 'would': "wouldn't", 'could': "couldn't", 'may': 'may not', 'might': 'might not', 'must': "mustn't"}
            return m.get(a.lower(), a + "n't")
        tag = f'{aux} it?' if neg_present else f'{neg_form(aux)} it?'
        if base.endswith('?'):
            base = base[:-1]
        return f'{base}, {tag}'
    except Exception:
        return text

def _to_negation(text: str) -> str:
    try:
        import warnings
        warnings.filterwarnings('ignore')
        import spacy
        try:
            from nltk.corpus import wordnet as wn
        except Exception:
            wn = None
        try:
            with SuppressStderr():
                nlp = spacy.load('en_core_web_sm')
        except Exception:
            return 'not ' + text
        doc = nlp(text)
        tokens = [t.text for t in doc]
        for (i, tok) in enumerate(doc):
            if tok.pos_ is None or not tok.pos_:
                continue
            if tok.pos_ in {'ADJ', 'ADV'} and tok.text and tok.text.strip():
                ant = None
                if wn is not None:
                    try:
                        for syn in wn.synsets(tok.text):
                            for l in syn.lemmas():
                                for a in l.antonyms():
                                    ant = a.name().replace('_', ' ')
                                    raise StopIteration
                    except StopIteration:
                        pass
                    except Exception:
                        continue
                if ant:
                    tokens[i] = f'not {ant}'
                else:
                    tokens[i] = f'not {tok.text}'
                return ' '.join(tokens)
        return 'not ' + text
    except Exception:
        return text

def _select_plan_for_style(plans: List[str], style: str) -> str:
    for p in plans:
        if style in p:
            return p
    return plans[0] if plans else style

def _build_abst_inputs(tok, x: str, y_label: Optional[int], *, target_ratio: float=0.5):
    try:
        toks = tok.tokenize(x)
    except Exception:
        toks = x.split()
    target_len = max(5, int(len(toks) * float(max(0.2, min(1.0, target_ratio)))))
    if isinstance(y_label, int):
        target_polarity = 'slightly upbeat' if y_label == 1 else 'slightly skeptical'
    else:
        target_polarity = 'neutral'
    prompt = f'<plan:abst>\nYou are a faithful abstractive rewriter.\nTask: Summarize the sentence into ONE sentence with the same factual meaning.\nConstraints:\n- Length: about {target_len} tokens.\n- Preserve named entities and numbers exactly.\n- Do NOT add new facts or opinions.\n- Keep the original stance unless unavoidable; prefer neutral phrasing.\n- Prefer concise wording that foregrounds {target_polarity} sentiment if already implied by the input, without introducing new claims.\nReturn ONLY the rewritten sentence.\nInput: "{x}"'
    enc = tok(prompt, return_tensors='pt', truncation=True, max_length=512)
    return enc

def _find_changed_word_spans(src: str, dst: str) -> Tuple[int, List[Tuple[int, int]]]:

    def _word_iters(text: str):
        return list(re.finditer('\\b\\w+\\b', text))
    a_it = _word_iters(src)
    b_it = _word_iters(dst)
    a_ws = [m.group(0) for m in a_it]
    b_ws = [m.group(0) for m in b_it]
    try:
        import difflib
        sm = difflib.SequenceMatcher(a=a_ws, b=b_ws)
        changed = -1
        for (tag, i1, i2, j1, j2) in sm.get_opcodes():
            if tag in ('replace', 'insert'):
                changed = j1 if j1 < len(b_it) else len(b_it) - 1
                break
        if changed < 0:
            return (-1, [])
        spans: List[Tuple[int, int]] = []
        for k in (changed - 1, changed, changed + 1):
            if 0 <= k < len(b_it):
                (s, e) = b_it[k].span()
                spans.append((int(s), int(e)))
        return (changed, spans)
    except Exception:
        return (-1, [])

def train_rl(args: RLArgs):
    import time
    if args.seed == 42:
        args.seed = int(time.time()) % 1000000
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f'[init] using random seed: {args.seed}')
    (train_split, val_split, text_field, label_field) = _load_task_split(args.dataset)
    try:
        if isinstance(args.subset_train, int) and args.subset_train > 0 and (len(train_split) > args.subset_train):
            train_split = train_split.select(range(int(args.subset_train)))
    except Exception:
        pass
    val_split_for_eval = val_split
    try:
        if isinstance(args.subset_val, int) and args.subset_val > 0 and (len(val_split) > args.subset_val):
            val_split_for_eval = val_split.select(range(int(args.subset_val)))
    except Exception:
        pass
    if str(args.dataset).lower() == 'amazon_polarity':
        try:

            def _mk_text(ex):
                t1 = str(ex.get('title', '')).strip()
                t2 = str(ex.get('content', '')).strip()
                s = (t1 + '. ' + t2).strip()
                if not s:
                    s = t1 or t2
                return {'text': s}
            train_split = train_split.map(_mk_text)
            val_split = val_split.map(_mk_text)
            val_split_for_eval = val_split_for_eval.map(_mk_text)
            text_field = 'text'
        except Exception:
            text_field = 'content'
    train_list = list(train_split)[:args.subset_train or None]
    val_list = list(val_split_for_eval)[:args.subset_val or None]
    extra_val_n = int(getattr(args, 'extra_val_for_train', 0) or 0)
    extra_val_appended = 0
    if extra_val_n > 0:
        try:
            extra_items = list(val_split)[:extra_val_n]
            train_list.extend(extra_items)
            extra_val_appended = len(extra_items)
        except Exception:
            extra_val_appended = 0
    victim = VictimProber(args.victim)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, trust_remote_code=True)
    ref_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, trust_remote_code=True)
    ref_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    ref_model.to(device)
    if extra_val_appended > 0:
        print(f'[init] extra_val_for_train: appended {extra_val_appended} validation samples (aligned with attack validation[:N])')
    print(f'[init] model loaded: {args.base_model} -> device={device}; train={len(train_list)} val={len(val_list)}')
    if getattr(args, 'gen_mode', False):
        prefer_gpt2_cpu(True)
        try:
            _ = compute_ppl('warmup')
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    num_steps = max(1, math.ceil(len(train_list) / max(1, args.batch_size)) * args.epochs)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=max(0, num_steps // 20), num_training_steps=num_steps)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    try:
        with open(Path(args.output_dir) / 'run_config.json', 'w', encoding='utf-8') as f:
            json.dump({'entrypoint': 'main.train', 'dataset': args.dataset, 'victim': to_portable_path(args.victim), 'victim_model_path': to_portable_path(resolve_victim_name(args.victim)), 'base_model': args.base_model, 'seed': args.seed, 'subset_train': args.subset_train, 'subset_val': args.subset_val, 'extra_val_for_train': int(getattr(args, 'extra_val_for_train', 0) or 0), 'extra_val_appended': int(extra_val_appended), 'train_size_effective': len(train_list), 'epochs': args.epochs, 'batch_size': args.batch_size, 'learning_rate': args.lr, 'beta_kl': args.beta_kl, 'reward_components': {'rew_flip': args.rew_flip, 'rew_drop': args.rew_drop, 'rew_sim_soft_w': args.rew_sim_soft_w, 'rew_ppl_soft_w': args.rew_ppl_soft_w, 'rew_det_soft_w': args.rew_det_soft_w, 'rew_edit_soft_w': args.rew_edit_soft_w, 'succ_aux_weight': args.succ_aux_weight, 'baseline_ema': args.baseline_ema}, 'constraints': {'train_sim': args.train_sim, 'train_ppl': args.train_ppl, 'train_det': args.train_det, 'train_edit_ratio_max': args.train_edit_ratio_max, 'sim_min': args.sim_min, 'ppl_max_ratio': args.ppl_max_ratio, 'edit_ratio_cap': args.edit_ratio_cap, 'budget_queries': args.budget_queries}, 'generation': {'gen_mode': args.gen_mode, 'samples_per_ex': args.samples_per_ex, 'rerank_pool': args.rerank_pool, 'gen_top_p': args.gen_top_p, 'gen_temperature': args.gen_temperature, 'gen_top_k': args.gen_top_k, 'max_new_tokens': args.max_new_tokens, 'ops_overlay': args.ops_overlay, 'ops_pool': args.ops_pool}}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f'[warn] failed to write run_config.json: {e}')
    b_global: float = 0.0
    plan_stat_sum: Dict[str, float] = {}
    plan_stat_cnt: Dict[str, int] = {}

    def _b(_: int) -> float:
        return float(b_global)

    def _update_baseline() -> Tuple[float, float]:
        nonlocal b_global
        model.eval()
        (asr, asr_before, total) = (0, 0, 0)
        plans = [p.strip() for p in (args.plan_templates or '').split(',') if p.strip()]
        if not plans:
            plans = ['noop+syn2+char_mid', 'neg+syn2+char_mid', 'llm+syn2+char_high']
        per_plan = max(1, int(math.ceil(args.val_total_candidates / float(max(1, len(plans))))))
        if 'tqdm' in globals() and tqdm is not None:
            _iter = enumerate(tqdm(val_list, desc='val', leave=False))
        else:
            _iter = enumerate(val_list)
        for (j, ex) in _iter:
            x = ex[text_field]
            y = int(ex[label_field])
            (y_hat_src, conf_src) = victim.predict(x)
            rewards: List[float] = []
            flipped = False
            flipped_before = False
            if args.gen_mode:
                ctrl = f'<e2e><sim≥{args.sim_min:.2f}><ppl≤{args.ppl_max_ratio:.1f}x><edit≤{int(args.edit_ratio_cap * 100)}%><budget:q{int(args.budget_queries)}><flip>'
                prompt = ctrl + '\n' + f'Rewrite into an adversarial variant while preserving meaning and fluency.\nInput: {x}\nOutput:'
                enc = tok(prompt, return_tensors='pt', truncation=True, max_length=512)
                enc = {k: v.to(device) for (k, v) in enc.items()}
                M = min(8, max(1, int(args.val_total_candidates)))
                M_pool = max(M, int(M * 3))
                with torch.no_grad():
                    outs = model.generate(**enc, do_sample=True, top_p=args.gen_top_p, temperature=args.gen_temperature, top_k=args.gen_top_k, num_return_sequences=M_pool, max_new_tokens=args.max_new_tokens)
                decs = [tok.decode(o, skip_special_tokens=True).strip() for o in (outs if isinstance(outs, list) else outs)]
                (uniq_all, seen) = ([], set())
                for s in decs:
                    if not s or s in seen:
                        continue
                    seen.add(s)
                    uniq_all.append(s)
                scored = []
                for s in uniq_all:
                    try:
                        edr = normalized_levenshtein(x, s)
                    except Exception:
                        edr = 1.0
                    scored.append((float(edr), s))
                scored.sort(key=lambda t: t[0])
                pick = [s for (_, s) in scored][:M]
                if not pick:
                    pick = uniq_all[:M] if uniq_all else [x]
                y_list = []
                for y_text in pick:
                    y_aug = y_text
                    if getattr(args, 'ops_overlay', False):
                        try:
                            (L0, L1) = _visible_span_by_offsets(y_aug, args.visible_L)
                        except Exception:
                            (L0, L1) = (0, min(len(y_aug), 128))
                        span = y_aug[L0:L1]
                        toks = [w for w in span.split() if any((c.isalpha() for c in w))]
                        if toks:
                            import random as _r
                            target = _r.choice(toks)
                            pool = [p.strip() for p in str(getattr(args, 'ops_pool', '')).split(',') if p.strip()]
                            if pool and target and (len(target) >= 3):
                                choice = _r.choice(pool)
                                (y2, ok) = apply_named(choice, y_aug, target=target)
                                if ok:
                                    y_aug = y2
                    y_list.append(y_aug)
                for y_text in y_list:
                    try:
                        (y_hat_new, _) = victim.predict(y_text)
                        p_new = victim.prob_of_label(y_text, y)
                        flipped_before |= y_hat_new != y
                        ok = _passes_all(x, y_text, args.train_sim, args.train_ppl, args.train_det, args.train_edit_ratio_max)
                        if ok:
                            flipped |= y_hat_new != y
                            p_src = victim.prob_of_label(x, y)
                            V = max(0.0, p_src - p_new)
                            r_raw = max(0.0, min(args.alpha_cap, args.eta * V))
                            cost = _excess_cost(x, y_text, lambda_c=args.lambda_c, lambda_w=args.lambda_w, lambda_o=args.lambda_o, allow_c=args.budget_allowed_chars, allow_w=args.budget_allowed_words, allow_o=args.budget_allowed_ops)
                            r = max(0.0, r_raw - cost)
                            rewards.append(r)
                    except Exception:
                        continue
                r_mean = float(sum(rewards) / max(1, len(rewards)))
                b_global = float(args.baseline_ema) * float(b_global) + (1.0 - float(args.baseline_ema)) * r_mean
                total += 1
                asr += int(flipped)
                asr_before += int(flipped_before)
                continue
            if args.staged_mode:

                def _visible_span_by_offsets(text: str, L: int) -> Tuple[int, int]:
                    enc = tok(text, return_offsets_mapping=True, return_tensors='pt', truncation=True)
                    offs = enc.get('offset_mapping')
                    if offs is None:
                        return (0, len(text))
                    ids = enc['input_ids'][0]
                    k = min(int(ids.shape[-1]), max(1, int(L)))
                    end = 0
                    for i2 in range(k):
                        (a, b) = offs[0][i2].tolist()
                        end = max(end, int(b))
                    return (0, max(1, min(end, len(text))))
                s1: List[Tuple[str, str]] = []
                s1.append(('noop', x))
                s1.append(('query', _to_query(x)))
                s1.append(('neg', _to_negation(x)))
                inp_s1 = _build_abst_inputs(tok, x, y_label=y, target_ratio=0.5)
                inp_s1 = {k: v.to(device) for (k, v) in inp_s1.items()}
                with torch.no_grad():
                    out_s1 = model.generate(**inp_s1, do_sample=True, top_p=args.gen_top_p, temperature=args.gen_temperature, max_new_tokens=args.max_new_tokens)
                s1.append(('abst', tok.decode(out_s1[0], skip_special_tokens=True).strip() or x))

                def _wn_synonyms_no_pos(word: str) -> List[str]:
                    try:
                        from nltk.corpus import wordnet as wn
                    except Exception:
                        return []
                    cands = []
                    try:
                        for s in wn.synsets(word):
                            for l in s.lemmas():
                                for a in l.antonyms():
                                    w = a.name().replace('_', ' ')
                                    if w.lower() != word.lower() and w not in cands:
                                        cands.append(w)
                        for s in wn.synsets(word):
                            for l in s.lemmas():
                                w = l.name().replace('_', ' ')
                                if w.lower() != word.lower() and w not in cands:
                                    cands.append(w)
                    except Exception:
                        return []
                    return cands[:8]

                def _match_case_simple(src: str, dst: str) -> str:
                    if not src:
                        return dst
                    if src.isupper():
                        return dst.upper()
                    if src[0].isupper():
                        return dst.capitalize()
                    return dst.lower()
                SENTIMENT_MAP = {'good': ['bad', 'awful', 'terrible', 'poor', 'lousy'], 'great': ['terrible', 'awful', 'dreadful'], 'excellent': ['awful', 'terrible', 'atrocious'], 'amazing': ['awful', 'terrible'], 'awesome': ['awful', 'terrible'], 'fantastic': ['awful', 'terrible'], 'superb': ['awful', 'terrible'], 'outstanding': ['awful', 'terrible'], 'brilliant': ['awful', 'terrible'], 'beautiful': ['ugly', 'awful'], 'entertaining': ['boring', 'dull'], 'engaging': ['boring', 'dull'], 'compelling': ['boring', 'dull'], 'hilarious': ['dull', 'unfunny'], 'funny': ['dull', 'unfunny'], 'fun': ['dull', 'boring'], 'enjoyable': ['boring', 'painful'], 'delightful': ['awful', 'dreary'], 'charming': ['annoying', 'awful'], 'heartwarming': ['heartless', 'cold'], 'moving': ['lifeless', 'flat'], 'inspiring': ['uninspiring', 'banal'], 'powerful': ['weak', 'flimsy'], 'clever': ['dumb', 'stupid'], 'smart': ['dumb', 'stupid'], 'original': ['derivative', 'cliched'], 'ambitious': ['aimless', 'lazy'], 'masterpiece': ['disaster', 'trainwreck'], 'must-see': ['skip', 'avoid'], 'fresh': ['stale', 'tired'], 'riveting': ['boring', 'dull'], 'gripping': ['boring', 'dull'], 'thrilling': ['tedious', 'boring'], 'positive': ['negative'], 'like': ['dislike'], 'love': ['hate'], 'enjoy': ['hate', 'dislike'], 'recommend': ['avoid', 'skip'], 'flawless': ['flawed'], 'bad': ['good', 'great'], 'awful': ['excellent', 'great', 'good'], 'terrible': ['great', 'excellent'], 'horrible': ['great', 'excellent'], 'dreadful': ['great', 'excellent'], 'atrocious': ['excellent', 'great'], 'lousy': ['good', 'great'], 'poor': ['good', 'strong'], 'weak': ['powerful', 'strong'], 'bland': ['flavorful', 'vivid'], 'boring': ['interesting', 'engaging'], 'dull': ['funny', 'engaging'], 'tedious': ['thrilling', 'engaging'], 'slow': ['fast', 'brisk'], 'confusing': ['clear', 'coherent'], 'messy': ['polished', 'coherent'], 'incoherent': ['coherent', 'clear'], 'predictable': ['surprising', 'original'], 'cliched': ['original', 'fresh'], 'cheesy': ['authentic', 'genuine'], 'lazy': ['ambitious', 'careful'], 'uneven': ['consistent', 'balanced'], 'pointless': ['meaningful', 'purposeful'], 'forgettable': ['memorable', 'unforgettable'], 'mediocre': ['excellent', 'great'], 'disappointing': ['satisfying', 'rewarding'], 'lifeless': ['moving', 'vibrant'], 'cringeworthy': ['hilarious', 'charming'], 'overlong': ['tight', 'concise'], 'overrated': ['underrated'], 'underrated': ['overrated'], 'flawed': ['flawless'], 'negative': ['positive'], 'hate': ['love'], 'dislike': ['like'], 'sad': ['happy']}

                def _replace_with_boundary(text: str, src: str, dst: str) -> str:
                    pattern = re.compile(f'\\b{re.escape(src)}\\b', flags=re.IGNORECASE)

                    def _sub(m):
                        return _match_case_simple(m.group(0), dst)
                    return pattern.sub(_sub, text, count=1)

                def _victim_top2_tokens(text_s1: str, y_true: int) -> List[str]:
                    toks = [w for w in text_s1.split() if any((c.isalpha() for c in w))]
                    scored: List[Tuple[float, str]] = []
                    for w in toks:
                        masked = _replace_with_boundary(text_s1, w, '')
                        try:
                            p_src = victim.prob_of_label(text_s1, y_true)
                            p_new = victim.prob_of_label(masked, y_true)
                            drop = max(0.0, p_src - p_new)
                        except Exception:
                            drop = 0.0
                        scored.append((drop, w))
                    scored.sort(key=lambda t: t[0], reverse=True)
                    out = []
                    seen = set()
                    for (_, w) in scored:
                        wl = w.lower()
                        if wl in seen:
                            continue
                        seen.add(wl)
                        out.append(w)
                        if len(out) >= 2:
                            break
                    if not out:
                        out = sorted(set(toks), key=lambda w: (-sum((c.isalpha() for c in w)), w.lower()))[:2]
                    return out

                def _s2_victim_driven(text_s1: str, y_true: int) -> List[str]:
                    targets = _victim_top2_tokens(text_s1, y_true)
                    out_variants: List[str] = []
                    for t2 in targets[:2]:
                        repls = _wn_synonyms_no_pos(t2)
                        if not repls:
                            repls = SENTIMENT_MAP.get(t2.lower(), [])
                        if not repls:
                            cand2 = _replace_with_boundary(text_s1, t2, f'not {t2}')
                        else:
                            cand2 = _replace_with_boundary(text_s1, t2, repls[0])
                        if cand2 != text_s1:
                            out_variants.append(cand2)
                    if not out_variants:
                        out_variants = [text_s1 + ' ', text_s1 + '  ']
                    elif len(out_variants) == 1:
                        out_variants.append(out_variants[0] + ' ')
                    return out_variants[:2]
                s2: List[Tuple[str, str]] = []
                for (s1_plan, base) in s1:
                    y_list2 = _s2_victim_driven(base, y)
                    for (j2, y2) in enumerate(y_list2[:2]):
                        s2.append((f'{s1_plan}+syn{j2 + 1}', y2))
                while len(s2) < 8:
                    s2.extend(s2[:max(0, 8 - len(s2))])
                s2 = s2[:8]

                def _char_perturb(text_s2: str, n: int) -> str:
                    try:
                        from .ops.char_misspell import get_key_neighbors
                        from .ops.char_homoglyph import LETTER_MAPPINGS
                    except Exception:
                        return text_s2
                    import random as _r
                    s_local = list(text_s2)
                    idxs = [i for (i, ch) in enumerate(s_local) if ch.isalpha()]
                    _r.shuffle(idxs)
                    idxs = idxs[:n]
                    for (k2, i2) in enumerate(idxs):
                        if k2 % 2 == 0:
                            kb = get_key_neighbors()
                            cand = kb.get(s_local[i2].lower())
                            if cand:
                                s_local[i2] = _r.choice(list(cand))
                        else:
                            repls = LETTER_MAPPINGS.get(s_local[i2], LETTER_MAPPINGS.get(s_local[i2].lower()))
                            if repls:
                                s_local[i2] = _r.choice(repls)
                    return ''.join(s_local)
                final: List[Tuple[str, str]] = []
                import random as _rand
                for (s2_plan, base2) in s2:
                    (L0, L1) = _visible_span_by_offsets(base2, args.visible_L)
                    Nvis = max(1, L1 - L0)
                    mid_n = min(args.s3_cap_mid, int(math.ceil(args.s3_mid_ratio * Nvis)))
                    high_n = min(args.s3_cap_high, int(math.ceil(args.s3_high_ratio * Nvis)))
                    vars_local: List[Tuple[str, str]] = []
                    if mid_n > 0:
                        for k3 in range(2):
                            _rand.seed(_rand.randint(0, 1000000))
                            vars_local.append((f'{s2_plan}+char_mid{k3 + 1}', _char_perturb(base2, mid_n)))
                    else:
                        vars_local.extend([(f'{s2_plan}+char_mid1', base2), (f'{s2_plan}+char_mid2', base2)])
                    if high_n > 0:
                        for k3 in range(2):
                            _rand.seed(_rand.randint(0, 1000000))
                            vars_local.append((f'{s2_plan}+char_high{k3 + 1}', _char_perturb(base2, high_n)))
                    else:
                        vars_local.extend([(f'{s2_plan}+char_high1', base2), (f'{s2_plan}+char_high2', base2)])
                    final.extend(vars_local)
                if len(final) > args.val_total_candidates:
                    final = final[:args.val_total_candidates]
                for (plan, cand) in final:
                    if not cand:
                        continue
                    (y_hat_new, _conf_new) = victim.predict(cand)
                    flipped_before |= y_hat_new != y
                    ok = _passes_all(x, cand, args.train_sim, args.train_ppl, args.train_det, args.train_edit_ratio_max)
                    if not ok:
                        rewards.append(0.0)
                        continue
                    flipped |= y_hat_new != y
                    V = 0.0
                    try:
                        p_src = victim.prob_of_label(x, y)
                        p_new = victim.prob_of_label(cand, y)
                        V = max(0.0, p_src - p_new)
                    except Exception:
                        pass
                    r_raw = max(0.0, min(args.alpha_cap, args.eta * V))
                    cost = _excess_cost(x, cand, lambda_c=args.lambda_c, lambda_w=args.lambda_w, lambda_o=args.lambda_o, allow_c=args.budget_allowed_chars, allow_w=args.budget_allowed_words, allow_o=args.budget_allowed_ops)
                    r = max(0.0, r_raw - cost)
                    rewards.append(r)
            else:
                for plan in plans:
                    for _ in range(per_plan):
                        inp = _build_inputs(tok, x, None, plan=plan, intensity=None)
                        inp = {k: v.to(device) for (k, v) in inp.items()}
                        with torch.no_grad():
                            out = model.generate(**inp, do_sample=True, top_p=args.gen_top_p, temperature=args.gen_temperature, max_new_tokens=args.max_new_tokens)
                        cand = tok.decode(out[0], skip_special_tokens=True).strip()
                        if not cand:
                            continue
                        (y_hat_new, _conf_new) = victim.predict(cand)
                        flipped_before |= y_hat_new != y
                        ok = _passes_all(x, cand, args.train_sim, args.train_ppl, args.train_det, args.train_edit_ratio_max)
                        if not ok:
                            rewards.append(0.0)
                            continue
                        flipped |= y_hat_new != y
                        try:
                            p_src = victim.prob_of_label(x, y)
                            p_new = victim.prob_of_label(cand, y)
                            V = max(0.0, p_src - p_new)
                        except Exception:
                            V = 0.0
                        r_raw = max(0.0, min(args.alpha_cap, args.eta * V))
                        cost = _excess_cost(x, cand, lambda_c=args.lambda_c, lambda_w=args.lambda_w, lambda_o=args.lambda_o, allow_c=args.budget_allowed_chars, allow_w=args.budget_allowed_words, allow_o=args.budget_allowed_ops)
                        r = max(0.0, r_raw - cost)
                        rewards.append(r)
            total += 1
            asr += int(flipped)
            asr_before += int(flipped_before)
        model.train()
        return (asr / max(1, total), asr_before / max(1, total))
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None
    for ep in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_list)
        (avg_loss, avg_r, steps) = (0.0, 0.0, 0)
        rng = range(0, len(train_list), max(1, args.batch_size))
        iterator = tqdm(rng, desc=f'epoch {ep}/{args.epochs}', leave=False) if tqdm else rng
        for i in iterator:
            batch = train_list[i:i + args.batch_size]
            if not batch:
                continue
            losses: List[torch.Tensor] = []
            rewards_f: List[float] = []
            for (j, ex) in enumerate(batch):
                x = ex[text_field]
                try:
                    _max_c = int(getattr(args, 'max_input_chars', 0) or 0)
                except Exception:
                    _max_c = 0
                if _max_c > 0 and isinstance(x, str) and (len(x) > _max_c):
                    x = x[:_max_c]
                y = int(ex[label_field])
                p_src = victim.prob_of_label(x, y)
                if args.gen_mode:
                    T = max(1, args.epochs)
                    t = (ep - 1) / T
                    sim_w = args.cur_sim_w0 * (1 - t) + args.cur_sim_w1 * t
                    ppl_w = args.cur_ppl_w0 * (1 - t) + args.cur_ppl_w1 * t
                    det_w = args.cur_det_w0 * (1 - t) + args.cur_det_w1 * t
                    edt_w = args.cur_edit_w0 * (1 - t) + args.cur_edit_w1 * t
                    ctrl = f'<e2e><sim≥{args.sim_min:.2f}><ppl≤{args.ppl_max_ratio:.1f}x><edit≤{int(args.edit_ratio_cap * 100)}%><budget:q{int(args.budget_queries)}><flip>'
                    prompt = ctrl + '\n' + f'Rewrite into an adversarial variant while preserving meaning and fluency.\nInput: {x}\nOutput:'
                    enc = tok(prompt, return_tensors='pt', truncation=True, max_length=512)
                    enc = {k: v.to(device) for (k, v) in enc.items()}
                    M = max(1, int(args.samples_per_ex))
                    M_pool = max(M, int(M * max(1, args.rerank_pool)))
                    with torch.no_grad():
                        outs = model.generate(**enc, do_sample=True, top_p=args.gen_top_p, temperature=args.gen_temperature, top_k=args.gen_top_k, num_return_sequences=M_pool, max_new_tokens=args.max_new_tokens)
                    decs = [tok.decode(o, skip_special_tokens=True).strip() for o in (outs if isinstance(outs, list) else outs)]
                    uniq_all = []
                    seen = set()
                    for s in decs:
                        if not s or s in seen:
                            continue
                        seen.add(s)
                        uniq_all.append(s)
                    scored = []
                    for s in uniq_all:
                        try:
                            edr = normalized_levenshtein(x, s)
                        except Exception:
                            edr = 1.0
                        scored.append((float(edr), s))
                    scored.sort(key=lambda t: t[0])
                    below = [s for (e, s) in scored if e <= float(args.edit_ratio_cap)]
                    above = [s for (e, s) in scored if e > float(args.edit_ratio_cap)]
                    pick = (below + above)[:M]
                    if not pick:
                        if uniq_all:
                            pick = uniq_all[:M]
                        else:
                            pick = [x]
                    cand_texts = pick
                    try:
                        src_ppl = compute_ppl(x)
                    except Exception:
                        src_ppl = None
                    for y_text in cand_texts:
                        y_aug = y_text
                        if getattr(args, 'ops_overlay', False):
                            try:
                                pool = [p.strip() for p in str(getattr(args, 'ops_pool', '')).split(',') if p.strip()]
                                target = None
                                if pool:
                                    try:
                                        (L0, L1) = _visible_span_by_offsets(y_aug, args.visible_L)
                                    except Exception:
                                        (L0, L1) = (0, min(len(y_aug), 128))
                                    span = y_aug[L0:L1]
                                    toks = [w for w in span.split() if any((c.isalpha() for c in w))]
                                    if toks:
                                        import random as _r
                                        target = _r.choice(toks)
                                if pool and target and (len(target) >= 3):
                                    import random as _r
                                    choice = _r.choice(pool)
                                    (y2, ok) = apply_named(choice, y_aug, target=target)
                                    if ok:
                                        y_aug = y2
                            except Exception:
                                pass
                        try:
                            (y_hat_new, _) = victim.predict(y_aug)
                            p_new = victim.prob_of_label(y_aug, y)
                            flipped = int(y_hat_new != y)
                            drop = max(0.0, p_src - p_new)
                        except Exception:
                            flipped = 0
                            drop = 0.0
                        try:
                            sim = compute_similarity(x, y_aug)
                        except Exception:
                            sim = 1.0
                        try:
                            pplr = compute_ppl_ratio(x, y_aug, src_ppl=src_ppl)
                        except Exception:
                            pplr = 1.0
                        try:
                            detv = proxy_score(y_aug)
                        except Exception:
                            detv = 0.0
                        try:
                            ed_ratio = normalized_levenshtein(x, y_aug)
                        except Exception:
                            ed_ratio = 0.0
                        try:
                            overflow = max(0.0, float(ed_ratio) - float(args.edit_ratio_cap))
                        except Exception:
                            overflow = 0.0
                        pen = 0.0
                        pen += sim_w * max(0.0, args.sim_min - float(sim))
                        pen += ppl_w * max(0.0, float(pplr) - args.ppl_max_ratio)
                        pen += det_w * float(detv)
                        pen += edt_w * float(ed_ratio)
                        pen += 2.0 * overflow
                        cost = _excess_cost(x, y_aug, lambda_c=args.lambda_c, lambda_w=args.lambda_w, lambda_o=args.lambda_o, allow_c=args.budget_allowed_chars, allow_w=args.budget_allowed_words, allow_o=args.budget_allowed_ops)
                        r_final = args.rew_flip * float(flipped) + args.rew_drop * float(drop) - float(pen) - float(cost)
                        logp_mean = _seq_logprob_mean(model, tok, enc, y_aug)
                        logr_mean = _seq_logprob_mean_ref(ref_model, tok, enc, y_aug)
                        dkl = logp_mean - float(logr_mean)
                        b = _b(i + j)
                        R = float(r_final) - b - args.beta_kl * dkl
                        loss = -(R * logp_mean)
                        losses.append(loss)
                        rewards_f.append(float(r_final))
                    continue
                layers = [max(1, int(x)) for x in (args.train_layers.split(',') if args.train_layers else ['1', '4', '8', '32'])]
                keep_fracs = [float(x) for x in (args.train_keep_fracs.split(',') if args.train_keep_fracs else ['1.0', '0.5', '0.25', '1.0'])]
                if args.staged_mode:
                    plans = []
                else:
                    plans_all = [p.strip() for p in (args.plan_templates or '').split(',') if p.strip()]
                    if args.plan_adaptive_routing and plan_stat_sum:
                        scores = {p: plan_stat_sum.get(p, 0.0) / max(1, plan_stat_cnt.get(p, 0)) for p in plans_all}
                        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        if args.train_top_plans > 0:
                            plans = [p for (p, _) in ranked[:args.train_top_plans]]
                        else:
                            plans = [p for (p, _) in ranked]
                    else:
                        plans = plans_all
                if not args.staged_mode:
                    candidates: List[Tuple[str, str]] = []
                    for plan in plans:
                        inp = _build_inputs(tok, x, None, plan=plan, intensity=None)
                        inp = {k: v.to(device) for (k, v) in inp.items()}
                        with torch.no_grad():
                            out = model.generate(**inp, do_sample=True, top_p=args.gen_top_p, temperature=args.gen_temperature, max_new_tokens=args.max_new_tokens)
                        y_text = tok.decode(out[0], skip_special_tokens=True).strip()
                        candidates.append((plan, y_text))
                    for (layer_idx, width) in enumerate(layers):
                        scored: List[Tuple[float, str, str]] = []
                        for (plan, y_text) in candidates:
                            if not _passes_all(x, y_text, args.train_sim, args.train_ppl, args.train_det, args.train_edit_ratio_max):
                                r_final = 0.0
                            else:
                                (y_hat_new, _) = victim.predict(y_text)
                                p_new = victim.prob_of_label(y_text, y)
                                V = max(0.0, p_src - p_new)
                                r_raw = max(0.0, min(args.alpha_cap, args.eta * V))
                                cost = _excess_cost(x, y_text, lambda_c=args.lambda_c, lambda_w=args.lambda_w, lambda_o=args.lambda_o, allow_c=args.budget_allowed_chars, allow_w=args.budget_allowed_words, allow_o=args.budget_allowed_ops)
                                r_final = max(0.0, r_raw - cost)
                                if args.succ_aux_weight > 0.0 and y_hat_new != y:
                                    r_final += args.succ_aux_weight
                            scored.append((float(r_final), plan, y_text))
                            inp = _build_inputs(tok, x, None, plan=plan, intensity=None)
                            inp = {k: v.to(device) for (k, v) in inp.items()}
                            logp_mean = _seq_logprob_mean(model, tok, inp, y_text)
                            logr_mean = _seq_logprob_mean_ref(ref_model, tok, inp, y_text)
                            dkl = logp_mean - float(logr_mean)
                            b = _b(i + j)
                            R = float(r_final) - b - args.beta_kl * dkl
                            loss = -(R * logp_mean)
                            losses.append(loss)
                            rewards_f.append(float(r_final))
                            if args.plan_mode and plan is not None:
                                plan_stat_sum[plan] = plan_stat_sum.get(plan, 0.0) + float(r_final)
                                plan_stat_cnt[plan] = plan_stat_cnt.get(plan, 0) + 1
                        keep_frac = keep_fracs[layer_idx] if layer_idx < len(keep_fracs) else 1.0
                        keep_n = max(1, int(math.ceil(len(scored) * keep_frac)))
                        scored.sort(key=lambda t: t[0], reverse=True)
                        parents = scored[:keep_n]
                        if layer_idx + 1 < len(layers):
                            next_candidates: List[Tuple[str, str]] = []
                            per_child = max(1, int(math.ceil(width / max(1, keep_n))))
                            for (r_final, plan, base) in parents:
                                for _k in range(per_child):
                                    inp = _build_inputs(tok, x, None, plan=plan, intensity=None)
                                    inp = {k: v.to(device) for (k, v) in inp.items()}
                                    with torch.no_grad():
                                        out = model.generate(**inp, do_sample=True, top_p=args.gen_top_p, temperature=args.gen_temperature, max_new_tokens=args.max_new_tokens)
                                    y_text2 = tok.decode(out[0], skip_special_tokens=True).strip()
                                    next_candidates.append((plan, y_text2))
                            candidates = next_candidates
                        else:
                            break
                else:
                    s1: List[Tuple[str, str]] = []
                    s1.append(('noop', x))
                    s1.append(('query', _to_query(x)))
                    s1.append(('neg', _to_negation(x)))
                    inp = _build_abst_inputs(tok, x, y_label=y, target_ratio=0.5)
                    inp = {k: v.to(device) for (k, v) in inp.items()}
                    with torch.no_grad():
                        out = model.generate(**inp, do_sample=True, top_p=args.gen_top_p, temperature=args.gen_temperature, max_new_tokens=args.max_new_tokens)
                    y1_abst = tok.decode(out[0], skip_special_tokens=True).strip()
                    s1.append(('abst', y1_abst))
                    _idf_cache: Dict[str, float] = {}

                    def _token_offsets(text: str) -> List[Tuple[int, int]]:
                        enc = tok(text, return_offsets_mapping=True, return_tensors='pt', truncation=True)
                        mapping = enc.get('offset_mapping')
                        if mapping is None:
                            return [(0, len(text))]
                        return [(int(a), int(b)) for (a, b) in mapping[0].tolist()]

                    def _visible_span_by_offsets(text: str, L: int) -> Tuple[int, int]:
                        enc = tok(text, return_offsets_mapping=True, return_tensors='pt', truncation=True)
                        input_ids = enc['input_ids'][0]
                        offs = enc['offset_mapping'][0]
                        n = int(input_ids.shape[-1])
                        k = min(n, max(1, int(L)))
                        end = 0
                        for i in range(k):
                            (a, b) = offs[i].tolist()
                            end = max(end, int(b))
                        return (0, max(1, min(end, len(text))))

                    def _idf(term: str) -> float:
                        t = term.lower()
                        v = _idf_cache.get(t)
                        if v is not None:
                            return v
                        try:
                            if not hasattr(_idf, 'built'):
                                df: Dict[str, int] = {}
                                total_docs = 0
                                for ex2 in train_list:
                                    total_docs += 1
                                    seen = set()
                                    for w in str(ex2.get(text_field, '')).split():
                                        wl = w.lower()
                                        if wl and wl not in seen and any((c.isalpha() for c in wl)):
                                            df[wl] = df.get(wl, 0) + 1
                                            seen.add(wl)
                                _idf.df = df
                                _idf.N = max(1, total_docs)
                                _idf.built = True
                            df = getattr(_idf, 'df', {})
                            N = int(getattr(_idf, 'N', 1))
                            d = max(1, int(df.get(t, 1)))
                            v = math.log((N + 1) / d)
                        except Exception:
                            v = 1.0
                        _idf_cache[t] = v
                        return v

                    def _auto_targets(text: str) -> List[str]:
                        (L0, L1) = _visible_span_by_offsets(text, args.visible_L)
                        span = text[L0:L1]
                        tokens = []
                        idx = 0
                        for w in text.split():
                            if any((c.isalpha() for c in w)):
                                pos = text.find(w, idx)
                                if pos == -1:
                                    continue
                                idx = pos + len(w)
                                if L0 <= pos < L1:
                                    tokens.append((w, pos))
                        if not tokens:
                            return []
                        center = (L0 + L1) / 2.0
                        scored: List[Tuple[float, str]] = []
                        for (w, p) in tokens:
                            idf = _idf(w)
                            dist = abs(p + len(w) / 2.0 - center) + 0.001
                            pos_w = 1.0 / dist
                            score = idf * pos_w
                            scored.append((score, w))
                        scored.sort(key=lambda t: t[0], reverse=True)
                        out = []
                        used = set()
                        for (_, w) in scored:
                            wl = w.lower()
                            if wl in used:
                                continue
                            used.add(wl)
                            out.append(w)
                            if len(out) >= max(1, args.s2_top_k):
                                break
                        return out
                    s2: List[Tuple[str, str]] = []
                    SENTIMENT_MAP_TR = {'good': ['mediocre', 'routine', 'bland', 'flat', 'thin', 'muted', 'plain', 'workmanlike', 'underwhelming', 'middling', 'safe', 'familiar', 'lightweight', 'derivative'], 'great': ['underwhelming', 'average', 'uneven', 'routine', 'middling', 'plain', 'muted'], 'excellent': ['overpraised', 'overrated', 'middling', 'uneven', 'routine'], 'amazing': ['underwhelming', 'middling', 'plain', 'muted'], 'awesome': ['middling', 'average', 'plain'], 'fantastic': ['underwhelming', 'bland', 'flat', 'routine'], 'superb': ['mediocre', 'thin', 'uneven'], 'outstanding': ['uneven', 'flat', 'muted'], 'brilliant': ['uneven', 'thin', 'clunky'], 'beautiful': ['bland', 'flat', 'plain'], 'entertaining': ['uneven', 'bland', 'thin'], 'engaging': ['flat', 'bland', 'shallow'], 'funny': ['flat', 'dry', 'thin'], 'enjoyable': ['so-so', 'thin', 'bland'], 'delightful': ['bland', 'tepid', 'slight'], 'moving': ['muted', 'flat', 'soft'], 'inspiring': ['muted', 'thin', 'soft'], 'powerful': ['thin', 'flat', 'soft'], 'smart': ['clumsy', 'uneven', 'muddy'], 'original': ['familiar', 'safe', 'derivative'], 'fresh': ['familiar', 'safe', 'routine'], 'riveting': ['bland', 'uneven', 'tepid'], 'thrilling': ['tepid', 'flat', 'low-stakes'], 'charming': ['slight', 'simple', 'cute'], 'gripping': ['uneven', 'bland', 'tepid'], 'hilarious': ['light', 'mild', 'dry'], 'witty': ['mild', 'light', 'dry'], 'clever': ['thin', 'clumsy', 'middling'], 'heartwarming': ['muted', 'soft', 'mild'], 'masterpiece': ['overrated', 'overpraised', 'uneven'], 'must-see': ['optional', 'minor', 'thin'], 'impressive': ['understated', 'quiet', 'mild'], 'stunning': ['muted', 'plain', 'low-key'], 'vibrant': ['muted', 'flat', 'low-key'], 'affecting': ['quiet', 'muted', 'soft'], 'poignant': ['quiet', 'soft', 'muted'], 'satisfying': ['modest', 'small', 'mild'], 'rewarding': ['modest', 'small', 'minor'], 'tight': ['small', 'simple', 'modest'], 'concise': ['small', 'simple', 'modest'], 'authentic': ['simple', 'plain', 'safe'], 'genuine': ['plain', 'simple', 'low-key'], 'memorable': ['minor', 'slight', 'light'], 'vivid': ['muted', 'flat', 'soft'], 'bad': ['okay', 'fine', 'decent', 'serviceable', 'passable'], 'awful': ['okay', 'decent', 'serviceable'], 'terrible': ['okay', 'fine', 'passable'], 'horrible': ['okay', 'passable', 'decent'], 'dreadful': ['okay', 'passable', 'decent'], 'atrocious': ['okay', 'decent'], 'lousy': ['okay', 'decent', 'fine'], 'poor': ['decent', 'solid', 'sound'], 'weak': ['solid', 'sound', 'steady'], 'bland': ['pleasant', 'mild', 'calm'], 'boring': ['engaging', 'interesting', 'steady'], 'dull': ['lively', 'engaging', 'bright'], 'tedious': ['brisk', 'snappy', 'tight'], 'slow': ['deliberate', 'measured', 'steady'], 'confusing': ['clear', 'coherent', 'tidy'], 'messy': ['tidy', 'coherent', 'neat'], 'incoherent': ['coherent', 'clear', 'tidy'], 'predictable': ['fresh', 'playful', 'surprising'], 'cliched': ['fresh', 'original', 'new'], 'cheesy': ['sincere', 'authentic', 'warm'], 'lazy': ['ambitious', 'careful', 'attentive'], 'uneven': ['balanced', 'consistent', 'steady'], 'pointless': ['meaningful', 'purposeful', 'focused'], 'forgettable': ['memorable', 'notable', 'distinct'], 'mediocre': ['decent', 'solid', 'capable'], 'disappointing': ['satisfying', 'rewarding', 'decent'], 'lifeless': ['lively', 'vibrant', 'warm'], 'cringeworthy': ['charming', 'light', 'playful'], 'overlong': ['tight', 'concise', 'brisk'], 'overrated': ['solid', 'worthy', 'decent'], 'flawed': ['solid', 'sound', 'polished'], 'noisy': ['clear', 'clean', 'focused'], 'chaotic': ['coherent', 'tidy', 'clear'], 'silly': ['playful', 'light', 'wry'], 'stupid': ['simple', 'straightforward', 'light'], 'dumb': ['simple', 'straightforward', 'light'], 'lame': ['light', 'gentle', 'simple'], 'annoying': ['light', 'mild', 'harmless'], 'tiresome': ['light', 'brisk', 'easy'], 'wooden': ['natural', 'relaxed', 'grounded'], 'stiff': ['natural', 'fluid', 'relaxed'], 'flat': ['calm', 'subtle', 'quiet'], 'shallow': ['light', 'breezy', 'simple'], 'contrived': ['neat', 'tidy', 'simple'], 'painful': ['moving', 'honest', 'stark'], 'angry': ['firm', 'direct', 'pointed'], 'harsh': ['firm', 'direct', 'stark'], 'grim': ['serious', 'sober', 'stern']}

                    def _match_case_simple_tr(src: str, dst: str) -> str:
                        if not src:
                            return dst
                        if src.isupper():
                            return dst.upper()
                        if src[0].isupper():
                            return dst.capitalize()
                        return dst.lower()

                    def _replace_with_boundary_tr(text: str, src: str, dst: str) -> str:
                        pattern = re.compile(f'\\b{re.escape(src)}\\b', flags=re.IGNORECASE)

                        def _sub(m):
                            return _match_case_simple_tr(m.group(0), dst)
                        return pattern.sub(_sub, text, count=1)
                    for (i2, (s1_plan, base)) in enumerate(s1):
                        targets = _auto_targets(base)
                        variants: List[str] = []
                        wn_out = []
                        for cand in wn_out or []:
                            if cand and cand != base and (cand not in variants):
                                variants.append(cand)
                            if len(variants) >= 2:
                                break
                        if len(variants) < 2:
                            for t in targets:
                                repls = SENTIMENT_MAP_TR.get(t.lower(), [])
                                if not repls:
                                    continue
                                cand = _replace_with_boundary_tr(base, t, repls[0])
                                if cand and cand != base and (cand not in variants):
                                    variants.append(cand)
                                if len(variants) >= 2:
                                    break
                        if len(variants) < 2:
                            for t in targets:
                                cand = _replace_with_boundary_tr(base, t, f'hardly {t}')
                                if cand and cand != base and (cand not in variants):
                                    variants.append(cand)
                                if len(variants) >= 2:
                                    break
                        if not variants:
                            variants = [base + ' ', base + '  ']
                        elif len(variants) == 1:
                            variants.append(variants[0] + ' ')
                        for (j, y2) in enumerate(variants[:2]):
                            plan_name = f'{s1_plan}+syn{j + 1}'
                            s2.append((plan_name, y2))
                    while len(s2) < 8:
                        s2.extend(s2[:max(0, 8 - len(s2))])
                    s2 = s2[:8]

                    def _visible_len(text: str, L: int) -> int:
                        return min(len(text), max(1, int(len(text) * 0.75)))

                    def _char_perturb(text: str, n: int) -> str:
                        try:
                            from .ops.char_misspell import get_key_neighbors
                            from .ops.char_homoglyph import LETTER_MAPPINGS
                        except Exception:
                            return text
                        import random
                        s = list(text)
                        idxs = [i for (i, ch) in enumerate(s) if ch.isalpha()]
                        random.shuffle(idxs)
                        idxs = idxs[:n]
                        for (k2, i2) in enumerate(idxs):
                            if k2 % 2 == 0:
                                kb = get_key_neighbors()
                                cand = kb.get(s[i2].lower())
                                if cand:
                                    s[i2] = random.choice(list(cand))
                            else:
                                repls = LETTER_MAPPINGS.get(s[i2], LETTER_MAPPINGS.get(s[i2].lower()))
                                if repls:
                                    s[i2] = random.choice(repls)
                        return ''.join(s)
                    final: List[Tuple[str, str]] = []
                    import random as _rand
                    for (s2_plan, base) in s2:
                        mid_n = 2
                        high_n = 3
                        variants = []
                        if mid_n > 0:
                            for k in range(2):
                                _rand.seed(_rand.randint(0, 1000000))
                                plan_name = f'{s2_plan}+char_mid{k + 1}'
                                variants.append((plan_name, _char_perturb(base, mid_n)))
                        else:
                            variants.extend([(f'{s2_plan}+char_mid1', base), (f'{s2_plan}+char_mid2', base)])
                        if high_n > 0:
                            for k in range(2):
                                _rand.seed(_rand.randint(0, 1000000))
                                plan_name = f'{s2_plan}+char_high{k + 1}'
                                variants.append((plan_name, _char_perturb(base, high_n)))
                        else:
                            variants.extend([(f'{s2_plan}+char_high1', base), (f'{s2_plan}+char_high2', base)])
                        final.extend(variants)
                    if len(final) > 32:
                        final = final[:32]
                    while len(final) < 32:
                        final.extend(final[:max(0, 32 - len(final))])
                    candidates = final
                    scored: List[Tuple[float, str, str]] = []
                    for (plan, y_text) in candidates:
                        if not _passes_all(x, y_text, args.train_sim, args.train_ppl, args.train_det, args.train_edit_ratio_max):
                            r_final = 0.0
                        else:
                            (y_hat_new, _) = victim.predict(y_text)
                            p_new = victim.prob_of_label(y_text, y)
                            V = max(0.0, p_src - p_new)
                            r_raw = max(0.0, min(args.alpha_cap, args.eta * V))
                            cost = _excess_cost(x, y_text, lambda_c=args.lambda_c, lambda_w=args.lambda_w, lambda_o=args.lambda_o, allow_c=args.budget_allowed_chars, allow_w=args.budget_allowed_words, allow_o=args.budget_allowed_ops)
                            r_final = max(0.0, r_raw - cost)
                            if args.succ_aux_weight > 0.0 and y_hat_new != y:
                                r_final += args.succ_aux_weight
                        scored.append((float(r_final), plan, y_text))
                        inp = _build_inputs(tok, x, None, plan=plan, intensity=None)
                        inp = {k: v.to(device) for (k, v) in inp.items()}
                        logp_mean = _seq_logprob_mean(model, tok, inp, y_text)
                        logr_mean = _seq_logprob_mean_ref(ref_model, tok, inp, y_text)
                        dkl = logp_mean - float(logr_mean)
                        b = _b(i + j)
                        R = float(r_final) - b - args.beta_kl * dkl
                        loss = -(R * logp_mean)
                        losses.append(loss)
                        rewards_f.append(float(r_final))
                        if args.plan_mode and plan is not None:
                            plan_stat_sum[plan] = plan_stat_sum.get(plan, 0.0) + float(r_final)
                            plan_stat_cnt[plan] = plan_stat_cnt.get(plan, 0) + 1
                    try:
                        topk = max(1, args.combo_topk)
                        tops = sorted(scored, key=lambda t: t[0], reverse=True)[:topk]
                        recs = [{'plan': p, 'score': float(s), 'text': y} for (s, p, y) in tops]
                        combo_path = Path(args.output_dir) / 'combos_scores.jsonl'
                        with open(combo_path, 'a', encoding='utf-8') as _f:
                            for r in recs:
                                _f.write(json.dumps(r, ensure_ascii=False) + '\n')
                    except Exception:
                        pass
            if not losses:
                continue
            loss_batch = torch.stack(losses).mean()
            loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            avg_loss += float(loss_batch.item())
            batch_r = float(sum(rewards_f) / max(1, len(rewards_f)))
            b_global = float(args.baseline_ema) * float(b_global) + (1.0 - float(args.baseline_ema)) * batch_r
            avg_r += batch_r
            steps += 1
            if steps % 100 == 0:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            if tqdm is None and steps % args.log_interval == 0 or (tqdm is not None and steps % args.log_interval == 0):
                print({'epoch': ep, 'step': steps, 'loss': round(avg_loss / max(1, steps), 4), 'avg_r': round(avg_r / max(1, steps), 4)})
        (val_asr, val_asr_before) = _update_baseline()
        print({'epoch': ep, 'loss': round(avg_loss / max(1, steps), 4), 'avg_r': round(avg_r / max(1, steps), 4), 'val_asr': round(val_asr * 100, 2), 'val_asr_before': round(val_asr_before * 100, 2)})
        ep_dir = Path(args.output_dir) / f'epoch_{ep}'
        ep_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ep_dir))
        tok.save_pretrained(str(ep_dir))
        if plan_stat_cnt:
            plan_scores = {p: plan_stat_sum[p] / max(1, plan_stat_cnt[p]) for p in plan_stat_sum}
            with open(Path(args.output_dir) / 'plan_scores.json', 'w', encoding='utf-8') as f:
                json.dump({'scores': plan_scores, 'counts': plan_stat_cnt}, f, ensure_ascii=False, indent=2)
            with open(ep_dir / 'plan_scores.json', 'w', encoding='utf-8') as f:
                json.dump({'scores': plan_scores, 'counts': plan_stat_cnt}, f, ensure_ascii=False, indent=2)
            top3 = sorted(plan_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print({'top_plans': top3})
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
if __name__ == '__main__':
    import argparse
    from dataclasses import asdict
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='', help='可选：YAML 配置文件，作为默认值。命令行可覆盖其中字段。')
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--dataset', default='sst2')
    ap.add_argument('--subset_train', type=int, default=2000)
    ap.add_argument('--subset_val', type=int, default=200)
    ap.add_argument('--extra_val_for_train', type=int, default=200, help='将 validation 前 N 条追加到训练集（与 attack 的 validation[:subset] 对齐）')
    ap.add_argument('--victim', default='distilbert-base-uncased-finetuned-sst-2-english')
    ap.add_argument('--base_model', default='google/flan-t5-base')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-05)
    ap.add_argument('--beta_kl', type=float, default=0.02)
    ap.add_argument('--eta', type=float, default=1.0)
    ap.add_argument('--alpha_cap', type=float, default=1.0)
    ap.add_argument('--gen_top_p', type=float, default=0.95)
    ap.add_argument('--gen_temperature', type=float, default=1.0)
    ap.add_argument('--gen_top_k', type=int, default=300)
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--sim', default='cos>=0.85')
    ap.add_argument('--ppl', default='<=1.3x')
    ap.add_argument('--det', default='<=0.6')
    ap.add_argument('--edit_ratio_max', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gen_mode', action='store_true', default=True)
    ap.add_argument('--samples_per_ex', type=int, default=36)
    ap.add_argument('--sim_min', type=float, default=0.8)
    ap.add_argument('--ppl_max_ratio', type=float, default=2.0)
    ap.add_argument('--budget_queries', type=int, default=40)
    ap.add_argument('--rerank_pool', type=int, default=3)
    ap.add_argument('--edit_ratio_cap', type=float, default=0.15)
    ap.add_argument('--rew_flip', type=float, default=2.3)
    ap.add_argument('--rew_drop', type=float, default=1.2)
    ap.add_argument('--rew_sim_soft_w', type=float, default=0.5)
    ap.add_argument('--rew_ppl_soft_w', type=float, default=0.25)
    ap.add_argument('--rew_det_soft_w', type=float, default=0.25)
    ap.add_argument('--rew_edit_soft_w', type=float, default=0.25)
    ap.add_argument('--cur_sim_w0', type=float, default=0.2)
    ap.add_argument('--cur_sim_w1', type=float, default=0.3)
    ap.add_argument('--cur_ppl_w0', type=float, default=0.05)
    ap.add_argument('--cur_ppl_w1', type=float, default=0.1)
    ap.add_argument('--cur_det_w0', type=float, default=0.1)
    ap.add_argument('--cur_det_w1', type=float, default=0.5)
    ap.add_argument('--cur_edit_w0', type=float, default=0.1)
    ap.add_argument('--cur_edit_w1', type=float, default=0.3)
    ap.add_argument('--adaptive_routing', action='store_true')
    ap.add_argument('--route_gamma', type=float, default=0.7)
    ap.add_argument('--route_temp', type=float, default=1.0)
    ap.add_argument('--succ_aux_weight', type=float, default=1.0)
    ap.add_argument('--train_samples_per_ex', type=int, default=2)
    ap.add_argument('--val_each_type', type=int, default=12)
    ap.add_argument('--budget_allowed_chars', type=int, default=5)
    ap.add_argument('--budget_allowed_words', type=int, default=4)
    ap.add_argument('--budget_allowed_ops', type=int, default=10)
    ap.add_argument('--budget_lambda', type=float, default=0.1)
    ap.add_argument('--lambda_c', type=float, default=0.01)
    ap.add_argument('--lambda_w', type=float, default=0.005)
    ap.add_argument('--lambda_o', type=float, default=0.005)
    ap.add_argument('--train_sim', default='cos>=0.65')
    ap.add_argument('--train_ppl', default='<=2.5x')
    ap.add_argument('--train_det', default='<=0.95')
    ap.add_argument('--train_edit_ratio_max', type=float, default=0.9)
    ap.add_argument('--gsr_threshold', type=float, default=0.1)
    ap.add_argument('--nmge_low_threshold', type=float, default=0.02)
    ap.add_argument('--staged_mode', action='store_true', default=True)
    ap.add_argument('--s1_total', type=int, default=4)
    ap.add_argument('--s2_top_k', type=int, default=2)
    ap.add_argument('--s2_per_target', type=int, default=2)
    ap.add_argument('--s3_mid_ratio', type=float, default=0.1)
    ap.add_argument('--s3_cap_mid', type=int, default=5)
    ap.add_argument('--s3_high_ratio', type=float, default=0.15)
    ap.add_argument('--s3_cap_high', type=int, default=8)
    ap.add_argument('--visible_L', type=int, default=384)
    ap.add_argument('--baseline_ema', type=float, default=0.9, help='全局EMA基线的衰减系数，接近1更平滑')
    ap.add_argument('--ops_overlay', action='store_true', default=True)
    ap.add_argument('--ops_pool', default='misspell,homoglyph,semantic,phonetic')
    ap.add_argument('--max_input_chars', type=int, default=1200, help='截断过长输入（Amazon/AGNews），0=不截断')
    ap.add_argument('--combo_topk', type=int, default=3)
    ap.add_argument('--plan_adaptive_routing', action='store_true')
    ap.add_argument('--plan_gamma', type=float, default=0.7)
    ap.add_argument('--plan_temp', type=float, default=1.0)
    ap.add_argument('--train_top_plans', type=int, default=0)
    ap.add_argument('--val_total_candidates', type=int, default=48)
    ap.add_argument('--train_sim_min_cos', type=float, default=None, help='语义相似度下界，如 0.60 将映射为 cos>=0.60')
    ap.add_argument('--train_ppl_max_ratio', type=float, default=None, help='PPL 比率上界，如 3.0 将映射为 <=3.0x')
    ap.add_argument('--train_det_max', type=float, default=None, help='检测器分数上界，如 0.95 将映射为 <=0.95')
    args = ap.parse_args()
    merged = asdict(RLArgs())
    if args.cfg:
        try:
            import yaml
            with open(args.cfg, 'r', encoding='utf-8') as f:
                y = yaml.safe_load(f) or {}
            if isinstance(y, dict):
                for (k, v) in y.items():
                    if k in merged and v is not None:
                        merged[k] = v
        except Exception as e:
            print(f'[warn] 读取配置文件失败（忽略并使用命令行/默认）：{e}')
    for k in merged.keys():
        if hasattr(args, k):
            cli_val = getattr(args, k)
            try:
                default_val = ap.get_default(k)
            except Exception:
                default_val = None
            if cli_val != default_val:
                merged[k] = cli_val
    for k in ['val_candidates', 'sim', 'ppl', 'det', 'edit_ratio_max', 'adaptive_routing', 'route_gamma', 'route_temp']:
        if k in merged:
            merged.pop(k, None)
    if getattr(args, 'train_sim_min_cos', None) is not None:
        merged['train_sim'] = f'cos>={float(args.train_sim_min_cos):.2f}'
    if getattr(args, 'train_ppl_max_ratio', None) is not None:
        merged['train_ppl'] = f'<={float(args.train_ppl_max_ratio):.1f}x'
    if getattr(args, 'train_det_max', None) is not None:
        merged['train_det'] = f'<={float(args.train_det_max):.2f}'
    from dataclasses import fields as _dc_fields
    for f in _dc_fields(RLArgs):
        (k, t) = (f.name, f.type)
        if k in merged:
            v = merged[k]
            try:
                if t is float and isinstance(v, str):
                    merged[k] = float(v)
                elif t is int and isinstance(v, str):
                    merged[k] = int(v)
                elif t is bool and isinstance(v, str):
                    merged[k] = v.lower() in {'1', 'true', 'yes', 'y', 't'}
            except Exception:
                pass

    if not merged.get('output_dir'):
        merged['output_dir'] = _auto_output_dir(merged.get('victim', args.victim), merged.get('dataset', args.dataset))
    cfg_args = RLArgs(**merged)
    print('[init] parsed config:', {k: merged[k] for k in ['dataset', 'subset_train', 'subset_val', 'extra_val_for_train', 'victim', 'base_model', 'output_dir', 'epochs', 'batch_size', 'train_sim', 'train_ppl', 'train_det', 'train_edit_ratio_max', 'val_total_candidates'] if k in merged})
    train_rl(cfg_args)
