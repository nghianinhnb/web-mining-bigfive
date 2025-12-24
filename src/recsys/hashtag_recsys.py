from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.text import extract_all_hashtags
from src.config import EMBEDDING_MODEL, TRAIT_NAMES


class HashtagRecommender:
    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.encoder = SentenceTransformer(embedding_model)
        self.global_hashtag_counts = None
        self.hashtag_embeddings = None
        self.all_hashtags = None

    def fit(
        self,
        df: pd.DataFrame,
        tweet_col: str = "tweets",
    ) -> "HashtagRecommender":
        all_hashtags = []
        for tweets in df[tweet_col]:
            if isinstance(tweets, list):
                for tweet in tweets:
                    all_hashtags.extend(extract_all_hashtags(tweet))
            else:
                all_hashtags.extend(extract_all_hashtags(tweets))

        self.global_hashtag_counts = Counter(all_hashtags)
        self.all_hashtags = list(self.global_hashtag_counts.keys())

        if self.all_hashtags:
            self.hashtag_embeddings = self.encoder.encode(self.all_hashtags)

        return self

    def recommend_popularity(
        self,
        exclude_hashtags: List[str] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, int]]:
        exclude_set = set(exclude_hashtags or [])
        sorted_hashtags = sorted(
            self.global_hashtag_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        recommendations = []
        for hashtag, count in sorted_hashtags:
            if hashtag.lower() not in exclude_set:
                recommendations.append((hashtag, count))
            if len(recommendations) >= top_k:
                break
        return recommendations

    def recommend_content(
        self,
        user_text: str,
        exclude_hashtags: List[str] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        if not self.all_hashtags or self.hashtag_embeddings is None:
            return []

        exclude_set = set(h.lower() for h in (exclude_hashtags or []))
        user_embedding = self.encoder.encode([user_text])
        similarities = cosine_similarity(user_embedding, self.hashtag_embeddings)[0]

        hashtag_scores = list(zip(self.all_hashtags, similarities))
        hashtag_scores.sort(key=lambda x: x[1], reverse=True)

        recommendations = []
        for hashtag, score in hashtag_scores:
            if hashtag.lower() not in exclude_set:
                recommendations.append((hashtag, float(score)))
            if len(recommendations) >= top_k:
                break
        return recommendations

    def recommend_personality_aware(
        self,
        user_text: str,
        user_traits: Dict[str, float],
        exclude_hashtags: List[str] = None,
        top_k: int = 10,
        personality_weight: float = 0.3,
    ) -> List[Tuple[str, float]]:
        content_recs = self.recommend_content(
            user_text,
            exclude_hashtags=exclude_hashtags,
            top_k=top_k * 3,
        )

        if not content_recs:
            return []

        trait_vector = np.array([user_traits.get(t, 0.5) for t in TRAIT_NAMES])
        trait_vector = (trait_vector - 0.5) * 2

        reranked = []
        for hashtag, content_score in content_recs:
            hashtag_sentiment = self._estimate_hashtag_personality(hashtag)
            personality_sim = cosine_similarity(
                [trait_vector], [hashtag_sentiment]
            )[0][0]

            final_score = (
                (1 - personality_weight) * content_score
                + personality_weight * personality_sim
            )
            reranked.append((hashtag, float(final_score)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def _estimate_hashtag_personality(self, hashtag: str) -> np.ndarray:
        positive_words = {"happy", "love", "fun", "great", "awesome", "amazing"}
        social_words = {"party", "friends", "social", "team", "together"}
        creative_words = {"art", "music", "creative", "design", "photo"}
        work_words = {"work", "goal", "success", "business", "professional"}
        calm_words = {"peace", "calm", "relax", "mindful", "yoga"}

        hashtag_lower = hashtag.lower()
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]

        if any(w in hashtag_lower for w in creative_words):
            scores[0] = 0.8
        if any(w in hashtag_lower for w in work_words):
            scores[1] = 0.8
        if any(w in hashtag_lower for w in social_words):
            scores[2] = 0.8
        if any(w in hashtag_lower for w in positive_words):
            scores[3] = 0.7
        if any(w in hashtag_lower for w in calm_words):
            scores[4] = 0.8

        return np.array(scores)


def prepare_user_hashtags(
    df: pd.DataFrame,
    tweet_col: str = "tweets",
    holdout_ratio: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)

    records = []
    for _, row in df.iterrows():
        user_id = row["user_id"]
        tweets = row[tweet_col]

        if isinstance(tweets, list):
            all_hashtags = []
            for tweet in tweets:
                all_hashtags.extend(extract_all_hashtags(tweet))
        else:
            all_hashtags = extract_all_hashtags(tweets)

        unique_hashtags = list(set(all_hashtags))

        if len(unique_hashtags) < 2:
            continue

        np.random.shuffle(unique_hashtags)
        split_idx = max(1, int(len(unique_hashtags) * (1 - holdout_ratio)))

        train_hashtags = unique_hashtags[:split_idx]
        test_hashtags = unique_hashtags[split_idx:]

        records.append({
            "user_id": user_id,
            "train_hashtags": train_hashtags,
            "test_hashtags": test_hashtags,
            "all_hashtags": unique_hashtags,
        })

    return pd.DataFrame(records)

