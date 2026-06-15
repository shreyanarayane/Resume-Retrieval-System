# RecruiterFit AI

CPU-only candidate ranking system for the Redrob hackathon. The submitted ranking path does not use hosted LLM APIs, GPUs, or network calls during ranking.

## Approach

1. Normalize company candidate profiles into one `profile_text` per candidate.
2. Convert nested skills into structured skill evidence using proficiency, duration, and endorsements.
3. Convert Redrob behavioral signals into an `activity_score`.
4. Embed candidate profiles locally with SentenceTransformers.
5. Cache candidate embeddings under `data/processed/`.
6. Retrieve the top candidate pool with FAISS.
7. Rank deterministically using semantic fit, structured skill strength, and Redrob activity.
8. Export exactly 100 rows in the required submission format.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional notebook support:

```bash
pip install -r requirements-notebook.txt
```

## Data

Place the company files in:

```text
data/raw/
```

Expected files:

```text
data/raw/candidates.jsonl
data/raw/job_description.docx
```

## Run

```bash
python -m src.main --candidates data/raw/candidates.jsonl --job data/raw/job_description.docx --output outputs/ranked_candidates.csv --top-k 1000
```

The first run may create cached embeddings. Later runs reuse the cache and are much faster.

## Output

The output CSV is:

```text
outputs/ranked_candidates.csv
```

Columns match the required spec:

```text
candidate_id,rank,score,reasoning
```

## Architecture

```text
candidates.jsonl
      ↓
preprocess schema + skills + Redrob signals
      ↓
profile_text + structured_score + activity_score
      ↓
local embeddings + cached .npy file
      ↓
FAISS retrieval top 1000
      ↓
deterministic hybrid ranker
      ↓
top 100 submission CSV
```

