#!/usr/bin/env python3
"""
Create CSV scoring sheet for explanation evaluation.

Creates a CSV file with samples split among team members for rating.
Each person rates 12-13 samples based on the rubric:
- Groundedness (1-5): Is it based on evidence?
- Helpfulness (1-5): Is the explanation easy to understand/useful?
- Consistency (1-5): Is it contradictory to predicted traits?
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.config import LABELS_DIR, PROCESSED_DIR, SEED
from src.utils.io import setup_logging
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Create CSV scoring sheet for explanation evaluation")
    parser.add_argument("--samples_file", type=str, default=None, 
                       help="Path to samples metadata CSV")
    parser.add_argument("--n_raters", type=int, default=4, 
                       help="Number of team members for rating")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output path for scoring sheet CSV")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    logger = setup_logging("create_explain_scoring_sheet")
    set_seed(args.seed)

    # Load samples metadata
    samples_file = Path(args.samples_file) if args.samples_file else PROCESSED_DIR / "explain_samples_metadata.csv"
    
    if not samples_file.exists():
        logger.error(f"Samples metadata file not found: {samples_file}")
        logger.info("Please run scripts/generate_explain_samples.py first")
        sys.exit(1)

    logger.info(f"Loading samples from {samples_file}...")
    samples_df = pd.read_csv(samples_file)

    n_samples = len(samples_df)
    n_raters = args.n_raters

    # Calculate samples per rater
    samples_per_rater = n_samples // n_raters
    remainder = n_samples % n_raters

    logger.info(f"Total samples: {n_samples}")
    logger.info(f"Number of raters: {n_raters}")
    logger.info(f"Samples per rater: {samples_per_rater} (+{remainder} extra)")

    # Assign samples to raters
    samples_df = samples_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    scoring_rows = []
    rater_idx = 0
    samples_assigned = 0

    for idx, row in samples_df.iterrows():
        # Assign to current rater
        current_rater = f"rater_{rater_idx + 1}"
        samples_for_current = samples_per_rater + (1 if rater_idx < remainder else 0)

        if samples_assigned >= samples_for_current:
            rater_idx += 1
            samples_assigned = 0
            if rater_idx >= n_raters:
                break
            current_rater = f"rater_{rater_idx + 1}"

        scoring_rows.append({
            "rater_id": current_rater,
            "sample_id": row["sample_id"],
            "user_id": row["user_id"],
            "sample_file": row["sample_file"],
            "predicted_open": row.get("predicted_open", ""),
            "predicted_conscientious": row.get("predicted_conscientious", ""),
            "predicted_extroverted": row.get("predicted_extroverted", ""),
            "predicted_agreeable": row.get("predicted_agreeable", ""),
            "predicted_stable": row.get("predicted_stable", ""),
            "groundedness": "",  # To be filled by rater (1-5)
            "helpfulness": "",   # To be filled by rater (1-5)
            "consistency": "",   # To be filled by rater (1-5)
            "notes": "",         # Optional notes
        })

        samples_assigned += 1

    scoring_df = pd.DataFrame(scoring_rows)

    # Save scoring sheet
    output_file = Path(args.output_file) if args.output_file else LABELS_DIR / "explain_ratings.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    scoring_df.to_csv(output_file, index=False)

    logger.info(f"\nSaved scoring sheet to {output_file}")
    logger.info(f"Total rows: {len(scoring_df)}")

    # Print summary by rater
    logger.info("\n=== Sample Distribution ===")
    for rater_id in sorted(scoring_df["rater_id"].unique()):
        count = len(scoring_df[scoring_df["rater_id"] == rater_id])
        logger.info(f"  {rater_id}: {count} samples")

    logger.info("\n=== Scoring Instructions ===")
    logger.info("Rate each sample on a scale of 1-5:")
    logger.info("  - Groundedness: Does the explanation reference evidence? (1=No, 5=Yes, well-supported)")
    logger.info("  - Helpfulness: Is the explanation easy to understand and useful? (1=No, 5=Very)")
    logger.info("  - Consistency: Does the explanation match the predicted traits? (1=Contradictory, 5=Consistent)")


if __name__ == "__main__":
    main()
