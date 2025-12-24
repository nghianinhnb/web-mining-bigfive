import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi

from src.config import BM25_INDEX_PATH
from src.utils.text import preprocess_tweet


class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.doc_mapping: List[Dict] = []
        self.corpus: List[List[str]] = []

    def build(
        self,
        documents: List[Dict],
        text_key: str = "text",
    ) -> "BM25Index":
        self.doc_mapping = documents
        self.corpus = [doc[text_key].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)
        return self

    def search(
        self,
        query: str,
        top_k: int = 10,
        user_id: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        if self.bm25 is None:
            raise ValueError("Index not built")

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        doc_scores = list(zip(range(len(self.doc_mapping)), scores))

        if user_id:
            doc_scores = [
                (idx, score)
                for idx, score in doc_scores
                if self.doc_mapping[idx].get("user_id") == user_id
            ]

        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_results = doc_scores[:top_k]

        return [(self.doc_mapping[idx], score) for idx, score in top_results]

    def save(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = BM25_INDEX_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "corpus": self.corpus,
                    "doc_mapping": self.doc_mapping,
                },
                f,
            )

    def load(self, path: Optional[Path] = None) -> "BM25Index":
        if path is None:
            path = BM25_INDEX_PATH
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.corpus = data["corpus"]
        self.doc_mapping = data["doc_mapping"]
        self.bm25 = BM25Okapi(self.corpus)
        return self


def build_tweet_index(df, max_tweets_per_user: int = 200) -> BM25Index:
    documents = []
    for _, row in df.iterrows():
        user_id = row["user_id"]
        tweets = row["tweets"]
        if isinstance(tweets, str):
            tweets = [tweets]
        for i, tweet in enumerate(tweets[:max_tweets_per_user]):
            documents.append({
                "doc_id": f"{user_id}_{i}",
                "user_id": user_id,
                "text": tweet,
                "tweet_idx": i,
            })
    index = BM25Index()
    index.build(documents, text_key="text")
    return index

