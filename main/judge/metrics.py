from __future__ import annotations
from typing import Dict, List

def compute_asr(total: int, success: int) -> float:
    return success / max(1, total) * 100.0

def compute_basic_report(total: int, success: int, queries_sum: int) -> Dict[str, float]:
    asr = compute_asr(total, success)
    return {'total_samples': float(total), 'success': float(success), 'ASR(%)': round(asr, 2), 'queries_sum': float(queries_sum), 'queries_avg': round(queries_sum / max(1, total), 2)}

def compute_nasr(asr_percent: float, detect_rate: float) -> float:
    return max(0.0, asr_percent * (1.0 - detect_rate))

def summarize_scores(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {}
    import numpy as np
    arr = np.array(scores, dtype=float)
    return {'mean': float(arr.mean()), 'max': float(arr.max()), 'p95': float(np.quantile(arr, 0.95)), 'p50': float(np.quantile(arr, 0.5)), 'p05': float(np.quantile(arr, 0.05))}

def compute_asr_multi(total_samples: int, multi_success_sum: int) -> float:
    return float(multi_success_sum) / max(1, int(total_samples)) * 100.0
