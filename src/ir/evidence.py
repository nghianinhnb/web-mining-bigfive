from typing import Dict, List, Optional
import pandas as pd

from src.config import TRAIT_NAMES, TOP_K_EVIDENCE
from src.ir.bm25 import BM25Index


TRAIT_QUERIES = {
    "open": "creative imaginative curious artistic intellectual open-minded adventurous",
    "conscientious": "organized responsible reliable hardworking disciplined careful punctual",
    "extroverted": "social outgoing talkative energetic party friends fun enthusiastic",
    "agreeable": "kind helpful cooperative friendly empathetic understanding supportive",
    "stable": "calm relaxed peaceful steady composed patient secure confident",
}

TRAIT_QUERIES_EXTENDED = {
    "open": [
        "I love exploring new ideas and concepts",
        "creative artistic imaginative innovative thinking",
        "curious about the world learning new things",
        "appreciate art beauty aesthetics culture",
        "open to new experiences and adventures",
    ],
    "conscientious": [
        "I always complete my tasks on time",
        "organized disciplined responsible reliable",
        "hardworking dedicated focused achievement",
        "planning ahead setting goals priorities",
        "attention to detail careful thorough",
    ],
    "extroverted": [
        "I love being around people and socializing",
        "party friends fun excitement energy",
        "talkative outgoing social gatherings",
        "meeting new people making friends",
        "active energetic enthusiastic lively",
    ],
    "agreeable": [
        "I care about others and their feelings",
        "kind helpful supportive compassionate",
        "cooperative friendly understanding empathetic",
        "getting along harmony peace relationships",
        "trust generous forgiving accepting",
    ],
    "stable": [
        "I feel calm and emotionally balanced",
        "relaxed peaceful steady composed",
        "handling stress well remaining calm",
        "confident secure positive optimistic",
        "not easily upset or worried anxious",
    ],
}


def retrieve_evidence_for_user(
    index: BM25Index,
    user_id: str,
    top_k: int = TOP_K_EVIDENCE,
    use_extended: bool = False,
) -> Dict[str, List[Dict]]:
    evidence = {}

    queries = TRAIT_QUERIES_EXTENDED if use_extended else TRAIT_QUERIES

    for trait in TRAIT_NAMES:
        if use_extended:
            all_results = []
            for query in queries[trait]:
                results = index.search(query, top_k=top_k * 2, user_id=user_id)
                all_results.extend(results)
            seen_texts = set()
            unique_results = []
            for doc, score in sorted(all_results, key=lambda x: x[1], reverse=True):
                if doc["text"] not in seen_texts:
                    seen_texts.add(doc["text"])
                    unique_results.append({"tweet": doc["text"], "score": score})
                if len(unique_results) >= top_k:
                    break
            evidence[trait] = unique_results
        else:
            query = queries[trait]
            results = index.search(query, top_k=top_k, user_id=user_id)
            evidence[trait] = [
                {"tweet": doc["text"], "score": score}
                for doc, score in results
            ]

    return evidence


def retrieve_all_evidence(
    index: BM25Index,
    user_ids: List[str],
    top_k: int = TOP_K_EVIDENCE,
) -> pd.DataFrame:
    records = []
    for user_id in user_ids:
        evidence = retrieve_evidence_for_user(index, user_id, top_k)
        for trait, tweets in evidence.items():
            for rank, item in enumerate(tweets):
                records.append({
                    "user_id": user_id,
                    "trait": trait,
                    "rank": rank + 1,
                    "tweet": item["tweet"],
                    "score": item["score"],
                })
    return pd.DataFrame(records)

