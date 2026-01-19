#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import LABELS_DIR, RESULTS_DIR, EVIDENCE_PATH
from src.utils.io import setup_logging, load_parquet
from src.ir.ir_eval import create_ir_labels_template, evaluate_ir


def cli_labeling(template_df: pd.DataFrame) -> pd.DataFrame:
    labels = template_df.copy()

    users = labels["user_id"].unique()
    traits = labels["trait"].unique()

    print("\n=== IR Relevance Labeling Tool ===")
    print("For each tweet, enter 1 if relevant to the trait, 0 otherwise.")
    print("Enter 'q' to quit and save progress.\n")

    for user_id in users:
        print(f"\n{'='*60}")
        print(f"User: {user_id}")
        print(f"{'='*60}")

        for trait in traits:
            print(f"\n--- Trait: {trait.upper()} ---")
            user_trait = labels[
                (labels["user_id"] == user_id) & (labels["trait"] == trait)
            ].sort_values("rank")

            for idx, row in user_trait.iterrows():
                print(f"\n[Rank {row['rank']}] {row['tweet'][:100]}...")
                while True:
                    response = input("Relevant? (1/0/q): ").strip().lower()
                    if response == "q":
                        return labels
                    elif response in ["0", "1"]:
                        labels.loc[idx, "relevant"] = int(response)
                        break
                    else:
                        print("Invalid input. Enter 1, 0, or q.")

    return labels


def main():
    parser = argparse.ArgumentParser(description="IR labeling tool for evidence evaluation")
    parser.add_argument("--n_users", type=int, default=20, help="Number of users to label")
    parser.add_argument("--mode", choices=["create", "label", "eval"], default="create")
    parser.add_argument("--input_file", type=str, default=None, help="Input labels file for eval mode")
    args = parser.parse_args()

    logger = setup_logging("ir_label_tool")

    labels_path = LABELS_DIR / "ir_labels.csv"

    if args.mode == "create":
        logger.info("Loading evidence data...")
        evidence_df = load_parquet(EVIDENCE_PATH)

        logger.info(f"Creating template for {args.n_users} users...")
        template = create_ir_labels_template(evidence_df, n_users=args.n_users)

        template.to_csv(labels_path, index=False)
        logger.info(f"Created template with {len(template)} records at {labels_path}")
        logger.info("Edit this file manually or use --mode label for interactive labeling.")

    elif args.mode == "label":
        logger.info("Loading evidence data...")
        evidence_df = load_parquet(EVIDENCE_PATH)

        template = create_ir_labels_template(evidence_df, n_users=args.n_users)
        labels = cli_labeling(template)

        labels.to_csv(labels_path, index=False)
        logger.info(f"Saved labels to {labels_path}")

    elif args.mode == "eval":
        input_path = Path(args.input_file) if args.input_file else labels_path

        if not input_path.exists():
            logger.error(f"Labels file not found: {input_path}")
            sys.exit(1)

        logger.info(f"Loading labels from {input_path}...")
        labels_df = pd.read_csv(input_path)

        labeled = labels_df[labels_df["relevant"].notna()]
        logger.info(f"Evaluating {len(labeled)} labeled records...")

        metrics = evaluate_ir(labeled, k=5)

        logger.info("\n=== IR Evaluation Results ===")
        for k, v in sorted(metrics.items()):
            logger.info(f"  {k}: {v:.4f}")

        results_df = pd.DataFrame([metrics])
        output_path = RESULTS_DIR / "ir_eval.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()

