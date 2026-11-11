"""
data_loader.py — Multi-seller knowledge base loading and vector store management.
"""

import json
import asyncio
from pathlib import Path
import numpy as np

from app.config import logger
import generate_data

# Ensure all seller KBs & embeddings exist before bootstrapping
for _sid in generate_data.SELLER_IDS:
    if not generate_data.kb_path(_sid).exists() or not generate_data.embed_path(_sid).exists():
        logger.info(f"[startup] KB/embeddings missing for '{_sid}' — generating...")
        generate_data.build_knowledge_base(_sid)

BASE_KBs: dict[str, dict] = {}
for _sid in generate_data.SELLER_IDS:
    with open(generate_data.kb_path(_sid), "r", encoding="utf-8") as _f:
        BASE_KBs[_sid] = json.load(_f)
logger.info(f"[startup] Loaded KBs for sellers: {list(BASE_KBs.keys())}")


class PlaybookVectorStore:
    def __init__(self, playbook: list[dict], npz_path: Path):
        self.playbook = playbook
        archive = np.load(npz_path, allow_pickle=True)
        self.tfidf_matrix = archive["tfidf_matrix"]
        self.vocab = archive["vocab"].tolist()
        self.vocab_idx = {tok: i for i, tok in enumerate(self.vocab)}
        self.idf = archive["idf"]
        self.has_dense = bool(archive["has_dense"][0])
        self.dense_matrix = archive["dense_matrix"] if self.has_dense else None
        logger.info(f"[rag] Loaded {npz_path.name} (tfidf={self.tfidf_matrix.shape}, has_dense={self.has_dense})")

    def retrieve_sync(self, query: str, top_k: int = 2) -> list[dict]:
        """Synchronous TF-IDF cosine similarity retrieval."""
        tokens = generate_data.tokenize_with_ngrams(query)
        q_tf = np.zeros(len(self.vocab), dtype=np.float32)
        for tok in tokens:
            if tok in self.vocab_idx:
                q_tf[self.vocab_idx[tok]] += 1.0
        q_vec = q_tf * self.idf
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
            scores = (self.tfidf_matrix @ q_vec).tolist()
        else:
            scores = [0.0] * len(self.playbook)
        ranked = sorted(zip(scores, self.playbook), key=lambda x: x[0], reverse=True)
        logger.info(f"[rag] Query='{query[:45]}' -> Top={[e['id'] for _, e in ranked[:top_k]]}")
        return [entry for _, entry in ranked[:top_k]]

    async def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.retrieve_sync, query, top_k)


vector_stores: dict[str, PlaybookVectorStore] = {}
for _sid in generate_data.SELLER_IDS:
    vector_stores[_sid] = PlaybookVectorStore(
        BASE_KBs[_sid]["playbook"], generate_data.embed_path(_sid)
    )
