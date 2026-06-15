import argparse
from pathlib import Path

from .io_utils import read_job_description
from .preprocess import (
    load_candidates,
    normalize_candidates,
    prefilter_raw_active_candidates,
    prefilter_raw_role_candidates,
)
from .retrieval import retrieve_top_k, save_processed
from .scoring import rank_candidates, save_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank candidates for a job description.")
    parser.add_argument("--candidates", required=True, help="Path to candidates CSV/JSON/JSONL file.")
    parser.add_argument("--job", required=True, help="Path to job description text file.")
    parser.add_argument("--output", default="outputs/ranked_candidates.csv", help="Output CSV path.")
    parser.add_argument(
        "--processed-output",
        default=None,
        help="Optional path for normalized candidate profiles. If omitted, raw candidates are not written.",
    )
    parser.add_argument(
        "--retrieved-output",
        default="outputs/retrieved_top_1000.csv",
        help="Where to save the retrieved top-k candidate pool before final ranking.",
    )
    parser.add_argument("--top-k", type=int, default=1000, help="Candidates to retrieve semantically.")
    parser.add_argument(
        "--tfidf-prefilter-k",
        type=int,
        default=3500,
        help="Candidates to keep after fast raw HashingVectorizer prefilter before normalization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_description = read_job_description(args.job)
    raw = load_candidates(args.candidates)
    raw = prefilter_raw_active_candidates(raw)
    raw = prefilter_raw_role_candidates(raw, job_description, top_n=args.tfidf_prefilter_k)
    candidates = normalize_candidates(raw)
    if args.processed_output:
        save_processed(candidates, args.processed_output)

    retrieved = retrieve_top_k(
        candidates,
        job_description,
        top_k=args.top_k,
        tfidf_prefilter_k=None,
    )
    retrieved_output_path = Path(args.retrieved_output)
    retrieved_output_path.parent.mkdir(parents=True, exist_ok=True)
    retrieved.to_csv(retrieved_output_path, index=False)
    print(f"Saved retrieved top {len(retrieved)} candidates to {retrieved_output_path}")

    ranked = rank_candidates(retrieved, job_description)

    output_path = Path(args.output)
    save_submission(ranked, str(output_path), top_n=100)
    print(f"Saved ranked candidates to {output_path}")


if __name__ == "__main__":
    main()
