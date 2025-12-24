from typing import Dict, List, Set
import numpy as np


def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    rec_k = [r for r, _ in recommended[:k]] if isinstance(recommended[0], tuple) else recommended[:k]
    hits = sum(1 for r in rec_k if r.lower() in {x.lower() for x in relevant})
    return hits / k


def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    if len(relevant) == 0:
        return 0.0
    rec_k = [r for r, _ in recommended[:k]] if isinstance(recommended[0], tuple) else recommended[:k]
    hits = sum(1 for r in rec_k if r.lower() in {x.lower() for x in relevant})
    return hits / len(relevant)


def ap_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    if len(relevant) == 0:
        return 0.0

    rec_k = [r for r, _ in recommended[:k]] if recommended and isinstance(recommended[0], tuple) else recommended[:k]
    relevant_lower = {x.lower() for x in relevant}

    score = 0.0
    num_hits = 0

    for i, rec in enumerate(rec_k):
        if rec.lower() in relevant_lower:
            num_hits += 1
            precision = num_hits / (i + 1)
            score += precision

    return score / min(len(relevant), k)


def map_at_k(
    all_recommended: List[List[str]],
    all_relevant: List[Set[str]],
    k: int,
) -> float:
    aps = []
    for recs, rels in zip(all_recommended, all_relevant):
        aps.append(ap_at_k(recs, rels, k))
    return np.mean(aps) if aps else 0.0


def evaluate_recommender(
    recommended_lists: List[List],
    relevant_sets: List[Set[str]],
    k_values: List[int] = [5, 10],
) -> Dict[str, float]:
    metrics = {}

    for k in k_values:
        precisions = []
        recalls = []
        aps = []

        for recs, rels in zip(recommended_lists, relevant_sets):
            if len(rels) > 0 and len(recs) > 0:
                precisions.append(precision_at_k(recs, rels, k))
                recalls.append(recall_at_k(recs, rels, k))
                aps.append(ap_at_k(recs, rels, k))

        if precisions:
            metrics[f"precision@{k}"] = np.mean(precisions)
            metrics[f"recall@{k}"] = np.mean(recalls)
            metrics[f"map@{k}"] = np.mean(aps)
        else:
            metrics[f"precision@{k}"] = 0.0
            metrics[f"recall@{k}"] = 0.0
            metrics[f"map@{k}"] = 0.0

    return metrics

