"""
Vector Database & Top-K Similarity Search Engine for Lung CT Scans.
Uses fastembed ONNX vision embedding with cosine distance search.
"""
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from fastembed import ImageEmbedding


class CTScanVectorStore:
    def __init__(self, index_dir: Path | str = "data/index", model_name: str = "Qdrant/resnet50-onnx"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.metadata_path = self.index_dir / "metadata.json"

        print(f"Loading Vision Embedding Model: {model_name}...")
        self.embedder = ImageEmbedding(model_name)
        self.embeddings: np.ndarray | None = None
        self.metadatas: list[dict[str, Any]] = []

        self.load_index()

    def load_index(self):
        """Load stored index from disk if present."""
        if self.embeddings_path.exists() and self.metadata_path.exists():
            self.embeddings = np.load(str(self.embeddings_path))
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadatas = json.load(f)
            print(f"Loaded existing index with {len(self.metadatas)} items.")
        else:
            self.embeddings = None
            self.metadatas = []

    def build_index_from_gallery(self, gallery_manifest_path: Path | str):
        """Index all images in gallery manifest."""
        gallery_manifest_path = Path(gallery_manifest_path)
        with open(gallery_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        image_paths = [item["image_path"] for item in manifest]
        print(f"Generating embeddings for {len(image_paths)} gallery images...")

        # Batch embed images
        raw_embs = list(self.embedder.embed(image_paths))
        emb_matrix = np.array(raw_embs, dtype=np.float32)

        # L2 Normalize for Cosine Similarity via dot product
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        norm_embs = emb_matrix / norms

        self.embeddings = norm_embs
        self.metadatas = manifest

        # Save to disk
        np.save(str(self.embeddings_path), self.embeddings)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f, indent=2)

        print(f"Indexed {len(self.metadatas)} scans into {self.index_dir}")

    def search(self, query_image_path: str | Path, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Search K-top similar CT scans for query image.
        Returns list of matched dicts with similarity score and metadata.
        """
        if self.embeddings is None or len(self.metadatas) == 0:
            raise ValueError("Vector store is empty! Build or load index first.")

        query_image_path = str(query_image_path)
        raw_query_emb = list(self.embedder.embed([query_image_path]))[0]
        query_norm = raw_query_emb / max(np.linalg.norm(raw_query_emb), 1e-10)

        # Cosine similarity via dot product against normalized gallery
        scores = np.dot(self.embeddings, query_norm)

        # Top-K indices
        top_k = min(top_k, len(scores))
        best_indices = np.argsort(-scores)[:top_k]

        results = []
        for rank, idx in enumerate(best_indices, start=1):
            meta = dict(self.metadatas[idx])
            meta["rank"] = rank
            meta["similarity_score"] = float(round(float(scores[idx]), 4))
            meta["similarity_percent"] = f"{meta['similarity_score'] * 100:.1f}%"
            results.append(meta)

        return results
