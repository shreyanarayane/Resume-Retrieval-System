from pathlib import Path

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
    
def retrieve_top_k(
    candidates: pd.DataFrame,
    job_description: str,
    top_k: int = 1000,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    tfidf_prefilter_k: int | None = 10000,
) -> pd.DataFrame:
    candidates = tfidf_prefilter_candidates(candidates, job_description, top_n=tfidf_prefilter_k)
    model = load_embedding_model(model_name)

    print(f"Embedding {len(candidates)} filtered candidate profiles.")
    candidate_embeddings = embed_texts(model, candidates["profile_text"].tolist())

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
