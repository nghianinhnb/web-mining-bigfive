#!/usr/bin/env python3
"""
Aggregate explanation evaluation ratings.

Computes mean/std by criteria (groundedness, helpfulness, consistency),
and optionally inter-rater agreement if multiple raters rate same samples.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter

from src.config import LABELS_DIR, RESULTS_DIR
from src.utils.io import setup_logging


def compute_inter_rater_agreement(ratings_df: pd.DataFrame, criterion: str) -> dict:
    """
    Compute inter-rater agreement for a given criterion.
    
    Returns dict with:
    - mean_agreement: Average pairwise agreement
    - fleiss_kappa: Fleiss' kappa if available (requires scipy 1.9+)
    - agreement_matrix: Pairwise agreement between raters
    """
    # Filter to samples rated by multiple raters
    multi_rated = ratings_df.groupby("sample_id").size()
    multi_rated_samples = multi_rated[multi_rated > 1].index

    if len(multi_rated_samples) == 0:
        return {
            "mean_agreement": None,
            "fleiss_kappa": None,
            "n_agreements": 0,
            "n_pairs": 0,
        }

    # Get ratings for multi-rated samples
    multi_ratings = ratings_df[
        (ratings_df["sample_id"].isin(multi_rated_samples)) &
        (ratings_df[criterion].notna())
    ][["sample_id", "rater_id", criterion]]

    # Compute pairwise agreement
    agreements = []
    sample_ids = multi_ratings["sample_id"].unique()

    for sample_id in sample_ids:
        sample_ratings = multi_ratings[multi_ratings["sample_id"] == sample_id][criterion].values
        if len(sample_ratings) < 2:
            continue

        # Pairwise comparisons
        for i in range(len(sample_ratings)):
            for j in range(i + 1, len(sample_ratings)):
                # Exact agreement
                if sample_ratings[i] == sample_ratings[j]:
                    agreements.append(1)
                else:
                    agreements.append(0)

    mean_agreement = np.mean(agreements) if agreements else None

    # Try to compute Fleiss' kappa if we have enough data
    fleiss_kappa = None
    try:
        # Reshape data for Fleiss kappa
        sample_groups = multi_ratings.groupby("sample_id")[criterion].apply(list)
        if len(sample_groups) > 0:
            max_raters = max(len(ratings) for ratings in sample_groups)
            # Pad to same length for all samples
            ratings_matrix = []
            for ratings in sample_groups:
                padded = list(ratings) + [np.nan] * (max_raters - len(ratings))
                ratings_matrix.append(padded)

            if len(ratings_matrix) > 1 and max_raters > 1:
                # Convert to categorical (1-5 scale)
                # Fleiss kappa expects integer categories
                ratings_array = np.array(ratings_matrix, dtype=float)
                # Count agreements for each category
                # Simplified: just use mean agreement for now
                # Full Fleiss kappa requires more complex calculation
                pass
    except Exception:
        pass

    return {
        "mean_agreement": mean_agreement,
        "fleiss_kappa": fleiss_kappa,
        "n_agreements": len([a for a in agreements if a == 1]),
        "n_pairs": len(agreements),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate explanation evaluation ratings")
    parser.add_argument("--ratings_file", type=str, default=None,
                       help="Path to ratings CSV file")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output path for summary CSV")
    parser.add_argument("--compute_agreement", action="store_true",
                       help="Compute inter-rater agreement (requires samples rated by multiple raters)")
    args = parser.parse_args()

    logger = setup_logging("aggregate_explain_ratings")

    # Load ratings
    ratings_file = Path(args.ratings_file) if args.ratings_file else LABELS_DIR / "explain_ratings.csv"

    if not ratings_file.exists():
        logger.error(f"Ratings file not found: {ratings_file}")
        logger.info("Please run scripts/create_explain_scoring_sheet.py and fill in ratings")
        sys.exit(1)

    logger.info(f"Loading ratings from {ratings_file}...")
    ratings_df = pd.read_csv(ratings_file)

    # Filter to rated samples
    criteria = ["groundedness", "helpfulness", "consistency"]
    rated_df = ratings_df[
        ratings_df[criteria].notna().any(axis=1)
    ].copy()

    if len(rated_df) == 0:
        logger.warning("No rated samples found. Please fill in ratings first.")
        sys.exit(1)

    logger.info(f"Found {len(rated_df)} rated samples")

    # Convert ratings to numeric
    for criterion in criteria:
        rated_df[criterion] = pd.to_numeric(rated_df[criterion], errors="coerce")

    # Compute statistics by criterion
    summary_rows = []

    for criterion in criteria:
        valid_ratings = rated_df[criterion].dropna()
        
        if len(valid_ratings) == 0:
            logger.warning(f"No valid ratings for {criterion}")
            continue

        mean_val = float(valid_ratings.mean())
        std_val = float(valid_ratings.std())
        median_val = float(valid_ratings.median())
        min_val = float(valid_ratings.min())
        max_val = float(valid_ratings.max())
        n_ratings = len(valid_ratings)

        summary_rows.append({
            "criterion": criterion,
            "mean": mean_val,
            "std": std_val,
            "median": median_val,
            "min": min_val,
            "max": max_val,
            "n_ratings": n_ratings,
        })

        logger.info(f"\n=== {criterion.upper()} ===")
        logger.info(f"  Mean: {mean_val:.3f}")
        logger.info(f"  Std:  {std_val:.3f}")
        logger.info(f"  Median: {median_val:.3f}")
        logger.info(f"  Range: [{min_val}, {max_val}]")
        logger.info(f"  N ratings: {n_ratings}")

    # Compute overall statistics
    all_ratings = []
    for criterion in criteria:
        valid = rated_df[criterion].dropna()
        all_ratings.extend(valid.tolist())

    if all_ratings:
        overall_mean = np.mean(all_ratings)
        overall_std = np.std(all_ratings)
        summary_rows.append({
            "criterion": "overall",
            "mean": overall_mean,
            "std": overall_std,
            "median": np.median(all_ratings),
            "min": np.min(all_ratings),
            "max": np.max(all_ratings),
            "n_ratings": len(all_ratings),
        })

        logger.info(f"\n=== OVERALL ===")
        logger.info(f"  Mean: {overall_mean:.3f}")
        logger.info(f"  Std:  {overall_std:.3f}")
        logger.info(f"  N ratings: {len(all_ratings)}")

    # Inter-rater agreement (if requested and applicable)
    if args.compute_agreement:
        logger.info("\n=== Inter-Rater Agreement ===")
        for criterion in criteria:
            agreement = compute_inter_rater_agreement(rated_df, criterion)
            if agreement["mean_agreement"] is not None:
                logger.info(f"\n{criterion.upper()}:")
                logger.info(f"  Mean pairwise agreement: {agreement['mean_agreement']:.3f}")
                logger.info(f"  Agreement pairs: {agreement['n_agreements']}/{agreement['n_pairs']}")

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    output_file = Path(args.output_file) if args.output_file else RESULTS_DIR / "explain_eval_summary.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_file, index=False)

    logger.info(f"\nSaved summary to {output_file}")


if __name__ == "__main__":
    main()
