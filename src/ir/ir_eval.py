from typing import Dict, List
import numpy as np
import pandas as pd


def precision_at_k(relevant: List[int], k: int = 5) -> float:
    if k == 0:
        return 0.0
    relevant_k = relevant[:k]
    return sum(relevant_k) / k


def dcg_at_k(relevant: List[int], k: int = 5) -> float:
    relevant_k = relevant[:k]
    gains = np.array(relevant_k)
    discounts = np.log2(np.arange(2, len(relevant_k) + 2))
    return np.sum(gains / discounts)


def ndcg_at_k(relevant: List[int], k: int = 5) -> float:
    dcg = dcg_at_k(relevant, k)
    ideal_relevant = sorted(relevant[:k], reverse=True)
    idcg = dcg_at_k(ideal_relevant, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_ir(
    labels_df: pd.DataFrame,
    k: int = 5,
) -> Dict[str, float]:
    metrics = {}

    for trait in labels_df["trait"].unique():
        trait_df = labels_df[labels_df["trait"] == trait]

        precisions = []
        ndcgs = []

        for user_id in trait_df["user_id"].unique():
            user_trait_df = trait_df[trait_df["user_id"] == user_id].sort_values("rank")
            relevant = user_trait_df["relevant"].tolist()

            p_k = precision_at_k(relevant, k)
            n_k = ndcg_at_k(relevant, k)

            precisions.append(p_k)
            ndcgs.append(n_k)

        if precisions:
            metrics[f"p@{k}_{trait}"] = np.mean(precisions)
            metrics[f"ndcg@{k}_{trait}"] = np.mean(ndcgs)

    all_precisions = []
    all_ndcgs = []

    for user_id in labels_df["user_id"].unique():
        for trait in labels_df["trait"].unique():
            user_trait_df = labels_df[
                (labels_df["user_id"] == user_id) & (labels_df["trait"] == trait)
            ].sort_values("rank")

            if len(user_trait_df) > 0:
                relevant = user_trait_df["relevant"].tolist()
                all_precisions.append(precision_at_k(relevant, k))
                all_ndcgs.append(ndcg_at_k(relevant, k))

    metrics[f"avg_p@{k}"] = np.mean(all_precisions) if all_precisions else 0.0
    metrics[f"avg_ndcg@{k}"] = np.mean(all_ndcgs) if all_ndcgs else 0.0

    return metrics


def create_ir_labels_template(
    evidence_df: pd.DataFrame,
    n_users: int = 20,
) -> pd.DataFrame:
    user_ids = evidence_df["user_id"].unique()[:n_users]
    template = evidence_df[evidence_df["user_id"].isin(user_ids)].copy()
    template["relevant"] = 0
    return template

