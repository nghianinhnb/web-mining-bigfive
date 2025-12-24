from pathlib import Path
from typing import Dict, List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd

from src.config import CHROMA_DIR, EMBEDDING_MODEL, TRAIT_NAMES


class ChromaUserStore:
    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.persist_dir = persist_dir or CHROMA_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self.encoder = SentenceTransformer(embedding_model)
        self.collection = None

    def build_collection(
        self,
        df: pd.DataFrame,
        collection_name: str = "users",
    ) -> None:
        try:
            self.client.delete_collection(collection_name)
        except ValueError:
            pass

        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for _, row in df.iterrows():
            user_id = row["user_id"]
            text = row["text_concat"]

            embedding = self.encoder.encode(text).tolist()

            metadata = {"user_id": user_id}
            for trait in TRAIT_NAMES:
                col = f"y_{trait}"
                if col in row:
                    metadata[col] = float(row[col])

            ids.append(user_id)
            documents.append(text[:10000])
            metadatas.append(metadata)
            embeddings.append(embedding)

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def load_collection(self, collection_name: str = "users") -> None:
        self.collection = self.client.get_collection(collection_name)

    def get_similar_users(
        self,
        target_text: str,
        top_n: int = 3,
    ) -> List[Dict]:
        if self.collection is None:
            self.load_collection()

        query_embedding = self.encoder.encode(target_text).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
            include=["documents", "metadatas", "distances"],
        )

        similar_users = []
        for i in range(len(results["ids"][0])):
            user_info = {
                "user_id": results["ids"][0][i],
                "document": results["documents"][0][i][:500],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            }

            traits = {}
            for trait in TRAIT_NAMES:
                col = f"y_{trait}"
                if col in user_info["metadata"]:
                    traits[trait] = user_info["metadata"][col]
            user_info["traits"] = traits

            similar_users.append(user_info)

        return similar_users


def build_chroma_store(df: pd.DataFrame, persist_dir: Optional[Path] = None) -> ChromaUserStore:
    store = ChromaUserStore(persist_dir=persist_dir)
    store.build_collection(df)
    return store

