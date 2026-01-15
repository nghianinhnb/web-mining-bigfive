from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.text import extract_hashtags, extract_all_hashtags
from src.config import EMBEDDING_MODEL, TRAIT_NAMES, TRAIT_COLS


class HashtagRecommender:
    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.encoder = SentenceTransformer(embedding_model)
        self.global_hashtag_counts = None
        self.hashtag_embeddings = None
        self.all_hashtags = None
        self.hashtag_personality_profiles = {}  # hashtag -> avg personality vector
        self.filtered_hashtags = []
        self.min_freq = 1
        self.cooccurrence_probs = {} # ht_A -> {ht_B -> prob}
        self.second_order_cooc = {} # ht_A -> {ht_C -> prob} via transitive relation

    def fit(
        self,
        df: pd.DataFrame,
        tweet_col: str = "tweets",
        trait_cols: List[str] = None,
        min_freq: int = 3,
    ) -> "HashtagRecommender":
        """Fit recommender on training data.
        
        Args:
            df: DataFrame with user tweets and optionally personality traits
            tweet_col: Column containing tweets (list of strings)
            trait_cols: Columns with personality traits for computing hashtag profiles
            min_freq: Minimum frequency for a hashtag to be recommendable (default 3)
        """
        if trait_cols is None:
            trait_cols = TRAIT_COLS
            
        all_hashtags = []
        hashtag_user_traits = {}  # hashtag -> list of user trait vectors
        
        for _, row in df.iterrows():
            tweets = row[tweet_col]
            if isinstance(tweets, list):
                user_hashtags = []
                for tweet in tweets:
                    user_hashtags.extend(extract_hashtags(tweet))
            else:
                user_hashtags = extract_hashtags(tweets)
            
            all_hashtags.extend(user_hashtags)
            
            # Collect user traits for each hashtag they used
            user_traits = []
            for col in trait_cols:
                if col in row and pd.notna(row[col]):
                    user_traits.append(float(row[col]))
                else:
                    user_traits.append(0.5)  # default neutral
            
            if user_traits and len(user_traits) == len(trait_cols):
                for ht in set(user_hashtags):  # unique hashtags per user
                    ht_lower = ht.lower()
                    if ht_lower not in hashtag_user_traits:
                        hashtag_user_traits[ht_lower] = []
                    hashtag_user_traits[ht_lower].append(user_traits)

        # Build Co-occurrence Matrix (Association Rules)
        pair_counts = {}
        item_counts = {}
        
        for _, row in df.iterrows():
            tweets = row[tweet_col]
            if isinstance(tweets, list):
                # Flatten hashtags per user (Basket Analysis)
                user_hashtags = []
                for tweet in tweets:
                    user_hashtags.extend(extract_hashtags(tweet))
                unique_tags = list(set(h.lower() for h in user_hashtags))
                
                for h in unique_tags:
                    item_counts[h] = item_counts.get(h, 0) + 1
                    
                for i in range(len(unique_tags)):
                    for j in range(len(unique_tags)):
                        if i == j: continue
                        h_a = unique_tags[i]
                        h_b = unique_tags[j]
                        if h_a not in pair_counts: pair_counts[h_a] = {}
                        pair_counts[h_a][h_b] = pair_counts[h_a].get(h_b, 0) + 1
        
        # Compute Probabilities P(B|A)
        self.cooccurrence_probs = {}
        for h_a, targets in pair_counts.items():
            self.cooccurrence_probs[h_a] = {}
            count_a = item_counts[h_a]
            for h_b, count_ab in targets.items():
                self.cooccurrence_probs[h_a][h_b] = count_ab / count_a

        # Compute Second-Order Co-occurrence P(C|A) = sum_B P(C|B) * P(B|A)
        # This captures transitive relationships: A -> B -> C
        self.second_order_cooc = {}
        for h_a in self.cooccurrence_probs:
            self.second_order_cooc[h_a] = {}
            for h_b, p_b_a in self.cooccurrence_probs[h_a].items():
                if h_b in self.cooccurrence_probs:
                    for h_c, p_c_b in self.cooccurrence_probs[h_b].items():
                        if h_c != h_a and h_c != h_b:  # Avoid self-loops
                            transitive_prob = p_b_a * p_c_b
                            self.second_order_cooc[h_a][h_c] = self.second_order_cooc[h_a].get(h_c, 0) + transitive_prob

        self.global_hashtag_counts = Counter(all_hashtags)
        self.all_hashtags = list(self.global_hashtag_counts.keys())
        
        # Filter to only recommend hashtags with sufficient frequency
        self.min_freq = min_freq
        self.filtered_hashtags = [
            ht for ht, cnt in self.global_hashtag_counts.items() 
            if cnt >= min_freq
        ]

        # Compute hashtag personality profiles as average of users who used each hashtag
        for ht, traits_list in hashtag_user_traits.items():
            if traits_list:
                self.hashtag_personality_profiles[ht] = np.mean(traits_list, axis=0)

        if self.filtered_hashtags:
            self.hashtag_embeddings = self.encoder.encode(self.filtered_hashtags)
        else:
            self.hashtag_embeddings = self.encoder.encode(self.all_hashtags)
            self.filtered_hashtags = self.all_hashtags

        return self

    def get_profiles_df(self) -> pd.DataFrame:
        """Export learned profiles to DataFrame."""
        if not self.filtered_hashtags:
            return pd.DataFrame()
            
        data = []
        for i, ht in enumerate(self.filtered_hashtags):
            row = {"hashtag": ht}
            
            # Embedding
            if self.hashtag_embeddings is not None and i < len(self.hashtag_embeddings):
                row["embedding"] = self.hashtag_embeddings[i].tolist()
                
            # Personality Profile
            if ht.lower() in self.hashtag_personality_profiles:
                profile = self.hashtag_personality_profiles[ht.lower()]
                for idx, trait in enumerate(TRAIT_NAMES):
                    row[f"profile_{trait}"] = float(profile[idx])
            
            data.append(row)
            
        return pd.DataFrame(data)

    def load_profiles(self, profiles_df: pd.DataFrame):
        """Load profiles from DataFrame."""
        if profiles_df.empty:
            return
            
        self.filtered_hashtags = profiles_df["hashtag"].tolist()
        self.all_hashtags = self.filtered_hashtags # Assume loaded usage is restricted to this set
        self.global_hashtag_counts = Counter(self.filtered_hashtags) # Dummy counts if needed
        
        # Load embeddings
        if "embedding" in profiles_df.columns:
            # Check if embedding is list or string
            first_emb = profiles_df.iloc[0]["embedding"]
            if isinstance(first_emb, str):
                # Need to parse if loaded from CSV/Parquet as string (though parquet handles lists usually)
                pass 
            
            embeddings = np.array(profiles_df["embedding"].tolist())
            self.hashtag_embeddings = embeddings
            
        # Load personality profiles
        self.hashtag_personality_profiles = {}
        trait_cols = [f"profile_{t}" for t in TRAIT_NAMES]
        
        for _, row in profiles_df.iterrows():
            ht = row["hashtag"]
            # Check if we have trait columns
            if all(c in row for c in trait_cols):
                vec = np.array([row[c] for c in trait_cols])
                self.hashtag_personality_profiles[ht.lower()] = vec


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
        user_history_hashtags: List[str] = None,
        exclude_hashtags: List[str] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Recommend hashtags based on content similarity.
        
        If user_history_hashtags is provided, uses average embedding of user's past hashtags (Tag-based Profile).
        Otherwise falls back to encoding user_text.
        """
        if not self.filtered_hashtags or self.hashtag_embeddings is None:
            return []

        exclude_set = set(h.lower() for h in (exclude_hashtags or []))
        
        # Strategy: Use history hashtags if available (stronger signal), else user text
        user_embedding = None
        if user_history_hashtags:
            # Filter history to known hashtags to get embeddings
            valid_history = [
                h for h in user_history_hashtags 
                if h in self.global_hashtag_counts # We only need them to exist in our vocab
            ]
            if valid_history:
                # We need embeddings for these specific hashtags.
                # Since self.hashtag_embeddings only covers filtered_hashtags, 
                # we might need to encode these on the fly if they are not in filtered set.
                # Optimization: For now just encode them.
                history_embeddings = self.encoder.encode(valid_history)
                user_embedding = np.mean(history_embeddings, axis=0)
        
        if user_embedding is None:
            user_embedding = self.encoder.encode([user_text])

        similarities = cosine_similarity(user_embedding.reshape(1, -1), self.hashtag_embeddings)[0]

        hashtag_scores = list(zip(self.filtered_hashtags, similarities))
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
        user_history_hashtags: List[str] = None,
        top_k: int = 10,
        alpha: float = 0.3,
        popularity_weight: float = 0.1,
        keyword_weight: float = 2.0,
        use_mmr: bool = True,
        mmr_lambda: float = 0.6,
        cooccurrence_weight: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Personality-aware hashtag recommendation with re-ranking.
        
        Args:
            user_text: User's concatenated tweets
            user_traits: Dict mapping trait names to scores (0-1)
            exclude_hashtags: Hashtags to exclude from recommendations
            user_history_hashtags: List of hashtags user has used (for content profiling)
            top_k: Number of recommendations to return
            alpha: Weight for personality similarity 
            popularity_weight: Weight for popularity boost (log-scale)
        
        Returns:
            List of (hashtag, score) tuples
        """
        # Get content-based candidates (20x to allow deep re-ranking)
        content_recs = self.recommend_content(
            user_text,
            user_history_hashtags=user_history_hashtags,
            exclude_hashtags=exclude_hashtags,
            top_k=top_k * 20,
        )

        if not content_recs:
            return []

        # Prepare keywords from tweet content for direct matching
        tweet_tokens = set(extract_hashtags(user_text.lower()))
        tweet_words = set(user_text.lower().split()) | set(h.lower() for h in (user_history_hashtags or []))

        # User personality vector
        trait_vector = np.array([user_traits.get(t, 0.5) for t in TRAIT_NAMES])

        reranked_candidates = []
        max_log_count = np.log(max(self.global_hashtag_counts.values())) if self.global_hashtag_counts else 1.0

        for hashtag, content_score in content_recs:
            hashtag_personality = self._get_hashtag_personality(hashtag)
            
            # 1. Personality Similarity
            personality_sim = cosine_similarity(
                [trait_vector], [hashtag_personality]
            )[0][0]

            # 2. Popularity Score
            count = self.global_hashtag_counts.get(hashtag, 0)
            pop_score = np.log(count + 1) / max_log_count if max_log_count > 0 else 0
            
            # 3. Keyword Match Boost
            keyword_bonus = 0.0
            if hashtag.lower() in tweet_words:
                keyword_bonus = 1.0
                
            # 4. Co-occurrence Score
            cooc_score = 0.0
            if user_history_hashtags and self.cooccurrence_probs:
                # Sum of P(candidate | history_tag) for all history tags
                # This naturally handles frequency: if history has many tags predicting candidate, score is high
                for hist_tag in user_history_hashtags:
                    hist_tag = hist_tag.lower()
                    if hist_tag in self.cooccurrence_probs:
                        # self.cooccurrence_probs[A][B] is P(B|A)
                        cooc_score += self.cooccurrence_probs[hist_tag].get(hashtag.lower(), 0.0)
                
                # Normalize? Maybe not, summation rewards "more evidence". 
                # But to keep scale reasonable (0-1 approx), we might want to average or cap.
                # Let's average to make it comparable to cosine sim.
                if len(user_history_hashtags) > 0:
                     cooc_score /= len(user_history_hashtags)

            # Combined base score
            # Note: cooccurrence_weight acts as a booster key
            base_score = (1 - alpha) * content_score + alpha * personality_sim
            final_score = base_score + (popularity_weight * pop_score) + (keyword_weight * keyword_bonus) + (cooccurrence_weight * cooc_score)
            
            reranked_candidates.append({
                "hashtag": hashtag,
                "score": float(final_score),
                "embedding": self.hashtag_embeddings[self.filtered_hashtags.index(hashtag)] if hashtag in self.filtered_hashtags else self.encoder.encode([hashtag])[0]
            })

        # Sort by initial score
        reranked_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply MMR if requested
        if use_mmr:
            return self._mmr_rerank(reranked_candidates, top_k, mmr_lambda)
            
        return [(c["hashtag"], c["score"]) for c in reranked_candidates[:top_k]]

    def _mmr_rerank(self, candidates: List[Dict], top_k: int, mmr_lambda: float) -> List[Tuple[str, float]]:
        """Maximal Marginal Relevance (MMR) re-ranking for diversity."""
        selected = []
        candidate_pool = candidates[:]
        
        while len(selected) < top_k and candidate_pool:
            best_score = -float('inf')
            best_candidate = None
            
            for candidate in candidate_pool:
                # Relevance part
                relevance = candidate["score"]
                
                # Diversity part (max similarity to already selected)
                max_sim = 0.0
                if selected:
                    # Determine similarity with selected items
                    # Using dot product as approximation for cosine sim (if normalized)
                    cand_emb = candidate["embedding"]
                    for sel in selected:
                        sim = np.dot(cand_emb, sel["embedding"]) / (np.linalg.norm(cand_emb) * np.linalg.norm(sel["embedding"]) + 1e-9)
                        if sim > max_sim:
                            max_sim = sim
                            
                # MMR Score = lambda * Relevance - (1-lambda) * Max_Similarity
                mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                candidate_pool.remove(best_candidate)
                
        return [(c["hashtag"], c["score"]) for c in selected]

    def _get_hashtag_personality(self, hashtag: str) -> np.ndarray:
        """Get personality profile for a hashtag.
        
        Uses learned profile if available, otherwise falls back to keyword heuristics.
        """
        ht_lower = hashtag.lower()
        
        # Use learned profile if available
        if ht_lower in self.hashtag_personality_profiles:
            return self.hashtag_personality_profiles[ht_lower]
        
        # Fallback to keyword-based estimation
        return self._estimate_hashtag_personality_keywords(hashtag)
    
    def _estimate_hashtag_personality_keywords(self, hashtag: str) -> np.ndarray:
        """Fallback keyword-based personality estimation for unknown hashtags."""
        positive_words = {"happy", "love", "fun", "great", "awesome", "amazing"}
        social_words = {"party", "friends", "social", "team", "together"}
        creative_words = {"art", "music", "creative", "design", "photo"}
        work_words = {"work", "goal", "success", "business", "professional"}
        calm_words = {"peace", "calm", "relax", "mindful", "yoga"}

        hashtag_lower = hashtag.lower()
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]

        if any(w in hashtag_lower for w in creative_words):
            scores[0] = 0.7
        if any(w in hashtag_lower for w in work_words):
            scores[1] = 0.7
        if any(w in hashtag_lower for w in social_words):
            scores[2] = 0.7
        if any(w in hashtag_lower for w in positive_words):
            scores[3] = 0.6
        if any(w in hashtag_lower for w in calm_words):
            scores[4] = 0.7

        return np.array(scores)

    def recommend_enhanced_cooc(
        self,
        user_history_hashtags: List[str],
        exclude_hashtags: List[str] = None,
        top_k: int = 10,
        second_order_weight: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """Enhanced co-occurrence recommendation using 1st and 2nd order transitions.
        
        Args:
            user_history_hashtags: User's historical hashtags
            exclude_hashtags: Hashtags to exclude
            top_k: Number of recommendations
            second_order_weight: Weight for second-order co-occurrence (0-1)
        """
        exclude_set = set(h.lower() for h in (exclude_hashtags or []))
        scores = {}
        
        for hist_tag in user_history_hashtags:
            hist_tag_lower = hist_tag.lower()
            
            # First-order co-occurrence
            if hist_tag_lower in self.cooccurrence_probs:
                for target, prob in self.cooccurrence_probs[hist_tag_lower].items():
                    if target not in exclude_set:
                        scores[target] = scores.get(target, 0) + prob
            
            # Second-order co-occurrence
            if hist_tag_lower in self.second_order_cooc:
                for target, prob in self.second_order_cooc[hist_tag_lower].items():
                    if target not in exclude_set:
                        scores[target] = scores.get(target, 0) + second_order_weight * prob
        
        # Normalize by history length
        if len(user_history_hashtags) > 0:
            for k in scores:
                scores[k] /= len(user_history_hashtags)
        
        # Sort and return top_k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_items

    @staticmethod
    def reciprocal_rank_fusion(
        rankings: List[List[str]],
        k: int = 60,
    ) -> List[Tuple[str, float]]:
        """Reciprocal Rank Fusion (RRF) for combining multiple ranking lists.
        
        RRF score for item = sum(1 / (k + rank_in_list_i)) for all lists
        
        Args:
            rankings: List of ranked item lists (each from a different model)
            k: Constant to prevent high scores for top ranks (default 60)
        
        Returns:
            List of (item, rrf_score) sorted by score descending
        """
        rrf_scores = {}
        
        for ranking in rankings:
            for rank, item in enumerate(ranking):
                item_lower = item.lower() if isinstance(item, str) else item[0].lower()
                rrf_scores[item_lower] = rrf_scores.get(item_lower, 0) + 1 / (k + rank + 1)
        
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items

    def recommend_rrf_ensemble(
        self,
        user_text: str,
        user_traits: Dict[str, float],
        user_history_hashtags: List[str],
        exclude_hashtags: List[str] = None,
        top_k: int = 10,
        models: List[str] = None,
    ) -> List[Tuple[str, float]]:
        """Ensemble recommendation using Reciprocal Rank Fusion.
        
        Combines rankings from multiple models:
        - content: Content-based
        - cooc: First-order co-occurrence  
        - enhanced_cooc: 1st + 2nd order co-occurrence
        - personality: Personality-aware hybrid
        
        Args:
            user_text: User's tweet text
            user_traits: User personality traits
            user_history_hashtags: User's historical hashtags
            exclude_hashtags: Hashtags to exclude
            top_k: Number of final recommendations
            models: Which models to include (default: all)
        """
        if models is None:
            models = ["content", "cooc", "enhanced_cooc", "personality"]
        
        rankings = []
        candidate_k = top_k * 5  # Get more candidates for fusion
        
        for model in models:
            if model == "content":
                recs = self.recommend_content(
                    user_text=user_text,
                    user_history_hashtags=user_history_hashtags,
                    exclude_hashtags=exclude_hashtags,
                    top_k=candidate_k
                )
                rankings.append([r[0] for r in recs])
                
            elif model == "cooc":
                # Simple first-order co-occurrence
                scores = {}
                for hist_tag in user_history_hashtags:
                    hist_tag_lower = hist_tag.lower()
                    if hist_tag_lower in self.cooccurrence_probs:
                        for target, prob in self.cooccurrence_probs[hist_tag_lower].items():
                            if target.lower() not in set(h.lower() for h in (exclude_hashtags or [])):
                                scores[target] = scores.get(target, 0) + prob
                sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:candidate_k]
                rankings.append([r[0] for r in sorted_recs])
                
            elif model == "enhanced_cooc":
                recs = self.recommend_enhanced_cooc(
                    user_history_hashtags=user_history_hashtags,
                    exclude_hashtags=exclude_hashtags,
                    top_k=candidate_k
                )
                rankings.append([r[0] for r in recs])
                
            elif model == "personality":
                recs = self.recommend_personality_aware(
                    user_text=user_text,
                    user_traits=user_traits,
                    user_history_hashtags=user_history_hashtags,
                    exclude_hashtags=exclude_hashtags,
                    top_k=candidate_k,
                    use_mmr=False  # Disable MMR for pure ranking
                )
                rankings.append([r[0] for r in recs])
        
        # Filter empty rankings
        rankings = [r for r in rankings if len(r) > 0]
        
        if not rankings:
            return []
        
        # Apply RRF
        fused = self.reciprocal_rank_fusion(rankings)
        return fused[:top_k]


def prepare_user_hashtags(
    df: pd.DataFrame,
    tweet_col: str = "tweets",
    holdout_ratio: float = 0.2,
    seed: int = 42,
    min_freq: int = 3,
    global_hashtag_counts: Counter = None,
) -> pd.DataFrame:
    """Prepare user hashtag splits for evaluation.
    
    Args:
        df: User DataFrame
        tweet_col: Column with tweets
        holdout_ratio: Ratio of hashtags to hold out for testing
        seed: Random seed
        min_freq: Minimum frequency for hashtags to be included in test set
        global_hashtag_counts: Pre-computed global hashtag counts (optional)
    """
    np.random.seed(seed)
    
    # Compute global hashtag counts if not provided
    if global_hashtag_counts is None:
        global_hashtag_counts = Counter()
        for _, row in df.iterrows():
            tweets = row[tweet_col]
            if isinstance(tweets, list):
                for tweet in tweets:
                    global_hashtag_counts.update(extract_hashtags(tweet))
            else:
                global_hashtag_counts.update(extract_hashtags(tweets))
    
    # Filter to popular hashtags
    popular_hashtags = {ht for ht, cnt in global_hashtag_counts.items() if cnt >= min_freq}

    records = []
    for _, row in df.iterrows():
        user_id = row["user_id"]
        tweets = row[tweet_col]

        if isinstance(tweets, list):
            all_hashtags = []
            for tweet in tweets:
                all_hashtags.extend(extract_hashtags(tweet))
        else:
            all_hashtags = extract_hashtags(tweets)

        # Filter to only popular hashtags
        user_popular = [ht for ht in set(all_hashtags) if ht.lower() in popular_hashtags]

        if len(user_popular) < 2:
            continue

        np.random.shuffle(user_popular)
        split_idx = max(1, int(len(user_popular) * (1 - holdout_ratio)))

        train_hashtags = user_popular[:split_idx]
        test_hashtags = user_popular[split_idx:]

        records.append({
            "user_id": user_id,
            "train_hashtags": train_hashtags,
            "test_hashtags": test_hashtags,
            "all_hashtags": user_popular,
        })

    return pd.DataFrame(records)

