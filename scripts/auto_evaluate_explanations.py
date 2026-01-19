#!/usr/bin/env python3
"""
Automatically evaluate explanations based on heuristics.
This creates ratings based on automated analysis of explanations.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.config import LABELS_DIR, RESULTS_DIR


def evaluate_groundedness(explanation: dict, evidence: dict) -> float:
    """Evaluate if explanation references evidence (1-5 scale)."""
    score = 2.0  # Base score
    
    # Check if explanation mentions evidence
    trait_explanations = explanation.get("trait_explanations", {})
    overall_summary = explanation.get("overall_summary", "")
    
    evidence_mentions = 0
    total_traits = len(trait_explanations)
    
    for trait, exp_text in trait_explanations.items():
        # Check if explanation contains evidence keywords or references
        if trait in evidence and evidence[trait]:
            # Check if explanation mentions specific evidence
            for ev_item in evidence[trait][:3]:  # Check top 3 evidence
                tweet_text = ev_item.get("tweet", "")[:50]  # First 50 chars
                if tweet_text.lower() in exp_text.lower() or any(
                    word in exp_text.lower() for word in tweet_text.lower().split()[:5]
                ):
                    evidence_mentions += 1
                    break
    
    # Score based on evidence mentions
    if total_traits > 0:
        mention_ratio = evidence_mentions / total_traits
        score = 2.0 + mention_ratio * 3.0  # Scale to 2-5
    
    # Check overall summary
    if "evidence" in overall_summary.lower() or "post" in overall_summary.lower():
        score += 0.5
    
    return min(5.0, max(1.0, score))


def evaluate_helpfulness(explanation: dict) -> float:
    """Evaluate if explanation is clear and useful (1-5 scale)."""
    score = 3.0  # Base score
    
    trait_explanations = explanation.get("trait_explanations", {})
    overall_summary = explanation.get("overall_summary", "")
    
    # Check length and detail
    avg_exp_length = np.mean([len(exp) for exp in trait_explanations.values()])
    if avg_exp_length > 100:
        score += 0.5
    if avg_exp_length > 150:
        score += 0.5
    
    # Check if explanations are specific (not generic)
    generic_words = ["based on", "reflects", "indicates", "derived from"]
    specific_count = 0
    for exp_text in trait_explanations.values():
        if any(word in exp_text.lower() for word in ["example", "specific", "shows", "demonstrates"]):
            specific_count += 1
    
    if len(trait_explanations) > 0:
        specificity = specific_count / len(trait_explanations)
        score += specificity * 1.0
    
    # Check overall summary quality
    if len(overall_summary) > 50:
        score += 0.5
    
    return min(5.0, max(1.0, score))


def evaluate_consistency(explanation: dict, predicted_traits: dict) -> float:
    """Evaluate if explanation aligns with predicted traits (1-5 scale)."""
    score = 4.0  # Base score (usually consistent)
    
    trait_explanations = explanation.get("trait_explanations", {})
    exp_traits = explanation.get("predicted_traits", predicted_traits)
    
    inconsistencies = 0
    
    for trait in predicted_traits:
        pred_score = predicted_traits.get(trait, 0.5)
        exp_score = exp_traits.get(trait, pred_score)
        
        # Check if scores match
        if abs(pred_score - exp_score) > 0.1:
            inconsistencies += 1
        
        # Check if explanation text matches score level
        exp_text = trait_explanations.get(trait, "")
        if pred_score > 0.6:
            # High score - should mention positive aspects
            if any(word in exp_text.lower() for word in ["low", "lack", "limited", "not"]):
                inconsistencies += 0.5
        elif pred_score < 0.4:
            # Low score - should mention limitations
            if any(word in exp_text.lower() for word in ["high", "strong", "very", "excellent"]):
                inconsistencies += 0.5
    
    # Penalize inconsistencies
    score -= min(2.0, inconsistencies * 0.5)
    
    return min(5.0, max(1.0, score))


def main():
    samples_path = LABELS_DIR / "explain_samples.json"
    ratings_path = LABELS_DIR / "explain_ratings.csv"
    
    if not samples_path.exists():
        print(f"Error: {samples_path} not found")
        print("Run: python scripts/create_explain_eval_set.py first")
        sys.exit(1)
    
    print("Loading explanation samples...")
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"Found {len(samples)} samples")
    print("Evaluating explanations...")
    
    ratings = []
    
    for sample in samples:
        explanation = sample["explanation"]
        evidence = sample["evidence"]
        predicted_traits = sample["predicted_traits"]
        
        # Evaluate each criterion
        groundedness = evaluate_groundedness(explanation, evidence)
        helpfulness = evaluate_helpfulness(explanation)
        consistency = evaluate_consistency(explanation, predicted_traits)
        
        ratings.append({
            "sample_id": sample["sample_id"],
            "user_id": sample["user_id"],
            "groundedness": round(groundedness, 2),
            "helpfulness": round(helpfulness, 2),
            "consistency": round(consistency, 2),
            "rater_name": "auto_eval",
            "notes": "Automated evaluation based on heuristics"
        })
    
    # Create DataFrame
    ratings_df = pd.DataFrame(ratings)
    
    # Save ratings
    ratings_df.to_csv(ratings_path, index=False)
    print(f"\nSaved {len(ratings)} ratings to {ratings_path}")
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Groundedness: {ratings_df['groundedness'].mean():.2f} ± {ratings_df['groundedness'].std():.2f}")
    print(f"Helpfulness: {ratings_df['helpfulness'].mean():.2f} ± {ratings_df['helpfulness'].std():.2f}")
    print(f"Consistency: {ratings_df['consistency'].mean():.2f} ± {ratings_df['consistency'].std():.2f}")
    
    print("\nNow run: python scripts/summarize_explain_eval.py")


if __name__ == "__main__":
    main()
