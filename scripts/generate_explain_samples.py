#!/usr/bin/env python3
"""
Generate explanation samples for evaluation.

Selects 50 user samples from test set or app demo, generates predictions,
retrieves evidence, generates explanations, and saves as JSON files.
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.config import (
    PROCESSED_DIR,
    SPLITS_DIR,
    MODELS_DIR,
    TRAIT_NAMES,
    TOP_K_EVIDENCE,
    SEED,
)
from src.utils.io import setup_logging, load_parquet, load_splits
from src.utils.seed import set_seed
from src.utils.text import preprocess_tweets
from src.models.tfidf_ridge import TfidfRidgeModel
from src.ir.bm25 import BM25Index
from src.ir.evidence import retrieve_evidence_for_user
from src.rag.explain import get_explainer


def main():
    parser = argparse.ArgumentParser(description="Generate explanation samples for evaluation")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to generate")
    parser.add_argument("--lang", type=str, default="en", help="Language code")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for JSON files")
    parser.add_argument("--samples_file", type=str, default=None, help="Path to save samples metadata CSV")
    args = parser.parse_args()

    logger = setup_logging("generate_explain_samples")
    set_seed(args.seed)

    # Setup paths
    data_path = PROCESSED_DIR / f"pan15_{args.lang}.parquet"
    splits_dir = SPLITS_DIR / args.lang
    output_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DIR / "explain_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {data_path}...")
    df = load_parquet(data_path)

    # Load splits to get test set
    test_user_ids = []
    if splits_dir.exists():
        splits = load_splits(splits_dir)
        test_user_ids = splits.get("test", [])
        logger.info(f"Found {len(test_user_ids)} test users in splits")
    else:
        logger.warning(f"Splits directory {splits_dir} not found, using all users")

    # Select sample users
    if test_user_ids:
        available_users = df[df["user_id"].isin(test_user_ids)]["user_id"].unique()
    else:
        available_users = df["user_id"].unique()

    if len(available_users) < args.n_samples:
        logger.warning(f"Only {len(available_users)} users available, using all")
        sample_user_ids = available_users.tolist()
    else:
        sample_user_ids = np.random.choice(
            available_users, size=args.n_samples, replace=False
        ).tolist()

    logger.info(f"Selected {len(sample_user_ids)} users for evaluation")

    # Load model
    logger.info("Loading prediction model...")
    model_paths = [
        MODELS_DIR / "baseline.joblib",
        MODELS_DIR / "baseline_en.joblib",
        MODELS_DIR / f"baseline_{args.lang}.joblib",
    ]
    model = None
    for path in model_paths:
        if path.exists():
            try:
                model = TfidfRidgeModel.load(path)
                logger.info(f"Loaded model from {path}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                continue

    if model is None:
        logger.error("Model not found. Please train a model first.")
        sys.exit(1)

    # Load BM25 index
    logger.info("Loading BM25 index...")
    index = BM25Index()
    try:
        index.load()
    except Exception as e:
        logger.error(f"Failed to load BM25 index: {e}")
        sys.exit(1)

    # Load explainer
    explainer = get_explainer(use_openai=False)  # Use rule-based by default

    # Generate samples
    logger.info("Generating explanation samples...")
    samples_metadata = []

    for idx, user_id in enumerate(sample_user_ids, 1):
        logger.info(f"Processing sample {idx}/{len(sample_user_ids)}: {user_id}")

        user_data = df[df["user_id"] == user_id].iloc[0]

        # Get user tweets
        tweets = user_data.get("tweets", [])
        if isinstance(tweets, str):
            tweets = [tweets]
        if not tweets:
            logger.warning(f"No tweets found for user {user_id}, skipping")
            continue

        # Preprocess tweets
        processed_tweets = preprocess_tweets(tweets)

        # Predict traits
        text_concat = " ".join(processed_tweets)
        predictions = model.predict(pd.Series([text_concat]))[0]
        predicted_traits = {
            trait: float(np.clip(predictions[i], 0, 1))
            for i, trait in enumerate(TRAIT_NAMES)
        }

        # Retrieve evidence
        evidence = retrieve_evidence_for_user(
            index, user_id, top_k=TOP_K_EVIDENCE
        )

        # Generate explanation
        explanation = explainer.explain(
            predicted_traits=predicted_traits,
            evidence=evidence,
        )

        # Create sample JSON
        sample_data = {
            "user_id": user_id,
            "sample_id": idx,
            "predicted_traits": predicted_traits,
            "evidence": evidence,
            "explanation": explanation,
        }

        # Save individual JSON file
        sample_file = output_dir / f"sample_{idx:03d}_{user_id}.json"
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)

        # Collect metadata
        samples_metadata.append({
            "sample_id": idx,
            "user_id": user_id,
            "sample_file": sample_file.name,
            "predicted_open": predicted_traits["open"],
            "predicted_conscientious": predicted_traits["conscientious"],
            "predicted_extroverted": predicted_traits["extroverted"],
            "predicted_agreeable": predicted_traits["agreeable"],
            "predicted_stable": predicted_traits["stable"],
        })

        logger.info(f"  Saved to {sample_file.name}")

    # Save metadata CSV
    samples_df = pd.DataFrame(samples_metadata)
    samples_file = Path(args.samples_file) if args.samples_file else PROCESSED_DIR / "explain_samples_metadata.csv"
    samples_df.to_csv(samples_file, index=False)
    logger.info(f"\nSaved samples metadata to {samples_file}")
    logger.info(f"Total samples generated: {len(samples_metadata)}")
    logger.info(f"All samples saved in: {output_dir}")


if __name__ == "__main__":
    main()
