# RecruiterFit AI

Recruiter AI is an intelligent candidate retrieval and ranking system that analyzes job descriptions and resumes to identify the most relevant talent. By combining semantic understanding, skill matching, and contextual reasoning, it delivers accurate, explainable, and efficient candidate recommendations while running entirely on local CPU resources without relying on external APIs during inference.

## Problem Statement

Recruiters review hundreds of profiles but still miss strong candidates because keyword filters cannot understand real fit.

The goal is to build an AI ranking system that understands the job description, evaluates the full candidate profile, and recommends the best-fit candidates using semantic relevance, career evidence, skill depth, and behavioral hiring signals.

## What Our Solution Do?

Read a job description and actually understand what the role needs — not just pull out words.
Look at the full picture — career history, skills, behavioral signals, platform activity — and figure out who genuinely fits.
Deliver a shortlist that a recruiter can trust.


## Approach

1. Load the company candidate file and job description.
2. Filter clearly inactive or unavailable candidates using Redrob signals.
3. Apply a fast role-specific prefilter using lexical similarity.
4. Normalize each candidate into one rich `profile_text`.
5. Convert nested skills into a `structured_score`.
6. Convert Redrob behavioral signals into an `activity_score`.
7. Generate local SentenceTransformer embeddings.
8. Retrieve the strongest candidate pool using FAISS.
9. Rank candidates using semantic fit, skill strength, Redrob activity, and JD-specific evidence.
10. Export exactly 100 rows in the required CSV format.

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
python -m src.main --candidates data/raw/candidates.jsonl --job data/raw/job_description.docx --output outputs/final_top_100_with_reasoning.csv --top-k 1000
```


## Output

The output CSV is:

```text
outputs/final_top_100_with_reasoning.csv
```

Columns match the required spec:

```text
candidate_id,rank,score,reasoning
```

## Architecture

```text
candidates.jsonl + job_description.docx
      ↓
schema-aware loading
      ↓
active candidate filter using Redrob signals
      ↓
fast role prefilter using HashingVectorizer
      ↓
profile_text + structured_score + activity_score
      ↓
SentenceTransformer embeddings
      ↓
FAISS semantic retrieval top-k
      ↓
deterministic hybrid ranker
      ↓
top-100 submission CSV
```
## Demo / Sandbox

A hosted demo is available on Hugging Face Spaces:
```markdown
Resume Retrieval System: https://huggingface.co/spaces/shreyanarayane/Resume-Retrieval-System
```
The demo allows users to run the ranking system on a small candidate sample and generate a ranked CSV output. The full challenge submission is reproduced locally using the command in the setup section.

## Main Files
```
src/main.py        production entry point
src/preprocess.py  data loading, normalization, Redrob scoring
src/retrieval.py   embeddings, cache, FAISS retrieval
src/scoring.py     hybrid ranking and submission writer
src/io_utils.py    job description reader
app.py             small-sample Gradio demo
```
