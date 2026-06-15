from pathlib import Path
import hashlib
import json

import faiss
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name, local_files_only=True)


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 128) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def candidate_cache_key(candidates: pd.DataFrame, model_name: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    for candidate_id, profile_text in zip(candidates["candidate_id"], candidates["profile_text"]):
        hasher.update(str(candidate_id).encode("utf-8", errors="ignore"))
        hasher.update(b"\0")
        hasher.update(str(profile_text).encode("utf-8", errors="ignore"))
        hasher.update(b"\0")
    return hasher.hexdigest()[:16]


def load_or_create_candidate_embeddings(
    candidates: pd.DataFrame,
    model: SentenceTransformer,
    model_name: str,
    cache_dir: str | Path = "data/processed",
) -> np.ndarray:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = candidate_cache_key(candidates, model_name)
    embeddings_path = cache_dir / f"candidate_embeddings_{key}.npy"
    metadata_path = cache_dir / f"candidate_embeddings_{key}.json"

    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        if embeddings.shape[0] == len(candidates):
            print(f"Loaded cached candidate embeddings: {embeddings_path}")
            return embeddings.astype("float32")

    print("Creating candidate embeddings. This is slow only the first time for this processed file.")
    embeddings = embed_texts(model, candidates["profile_text"].tolist())
    np.save(embeddings_path, embeddings)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "candidate_count": len(candidates),
                "embedding_shape": list(embeddings.shape),
                "cache_key": key,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved candidate embeddings cache: {embeddings_path}")
    return embeddings


def retrieve_top_k(
    candidates: pd.DataFrame,
    job_description: str,
    top_k: int = 1000,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: str | Path = "data/processed",
    tfidf_prefilter_k: int | None = 10000,
) -> pd.DataFrame:
    candidates = tfidf_prefilter_candidates(candidates, job_description, top_n=tfidf_prefilter_k)
    model = load_embedding_model(model_name)
    candidate_embeddings = load_or_create_candidate_embeddings(candidates, model, model_name, cache_dir)
    index = build_faiss_index(candidate_embeddings)
    job_embedding = embed_texts(model, [job_description], batch_size=1)
    scores, indices = index.search(job_embedding, min(top_k, len(candidates)))

    retrieved = candidates.iloc[indices[0]].copy()
    retrieved["semantic_score"] = scores[0]
    return retrieved.reset_index(drop=True)


def tfidf_prefilter_candidates(
    candidates: pd.DataFrame,
    job_description: str,
    top_n: int | None = 10000,
) -> pd.DataFrame:
    if top_n is None or top_n <= 0 or len(candidates) <= top_n:
        out = candidates.copy()
        out["tfidf_score"] = 1.0
        return out.reset_index(drop=True)

    texts = candidates["profile_text"].fillna("").astype(str).tolist()
    vectorizer = HashingVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        n_features=2**18,
        alternate_sign=False,
        norm="l2",
    )
    matrix = vectorizer.transform([job_description] + texts)
    scores = (matrix[1:] @ matrix[0].T).toarray().ravel()
    keep = np.argsort(scores)[-top_n:][::-1]

    out = candidates.iloc[keep].copy().reset_index(drop=True)
    out["tfidf_score"] = scores[keep]
    print(f"TF-IDF prefilter kept {len(out)} of {len(candidates)} candidates")
    return out


def save_processed(candidates: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(path, index=False)
