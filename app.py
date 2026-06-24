from pathlib import Path

import gradio as gr

from src.io_utils import read_job_description
from src.preprocess import (
    load_candidates,
    normalize_candidates,
    prefilter_raw_active_candidates,
    prefilter_raw_role_candidates,
)
from src.retrieval import retrieve_top_k
from src.scoring import finalize, rank_candidates


DEFAULT_SAMPLE_CANDIDATES = Path("work/sample_candidates.json")
DEFAULT_FULL_CANDIDATES = Path("data/raw/candidates.jsonl")
DEFAULT_JOB_FILE = Path("data/raw/job_description.docx")
BACKEND_RETRIEVAL_K = 1000


def read_uploaded_or_default_candidates(candidate_file) -> Path:
    if candidate_file is not None:
        return Path(getattr(candidate_file, "name", candidate_file))
    if DEFAULT_FULL_CANDIDATES.exists():
        return DEFAULT_FULL_CANDIDATES
    return DEFAULT_SAMPLE_CANDIDATES


def resolve_job_text(job_text: str) -> str:
    if job_text and job_text.strip():
        return job_text.strip()
    if DEFAULT_JOB_FILE.exists():
        return read_job_description(DEFAULT_JOB_FILE)
    return (
        "Rank candidates for a role requiring relevant skills, proven work history, "
        "recent platform activity, and strong recruiter availability signals."
    )


def rank_candidates_ui(job_text, candidate_file, top_n):
    job_description = resolve_job_text(job_text)
    candidate_path = read_uploaded_or_default_candidates(candidate_file)

    raw = load_candidates(candidate_path)
    raw_count = len(raw)

    raw = prefilter_raw_active_candidates(raw)
    active_count = len(raw)

    raw = prefilter_raw_role_candidates(raw, job_description, top_n=min(3500, len(raw)))
    role_count = len(raw)

    candidates = normalize_candidates(raw)

    retrieved = retrieve_top_k(
        candidates,
        job_description,
        top_k=min(BACKEND_RETRIEVAL_K, len(candidates)),
        tfidf_prefilter_k=3500,
    )

    ranked = rank_candidates(retrieved, job_description)

    output_n = min(int(top_n), len(ranked), 100)
    demo_output = finalize(ranked).head(output_n)

    output_path = Path("outputs/demo_ranked_candidates.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    demo_output.to_csv(output_path, index=False, encoding="utf-8")

    status = (
        f"Candidate source: {candidate_path}\n"
        f"Raw candidates: {raw_count}\n"
        f"After active filter: {active_count}\n"
        f"After role prefilter: {role_count}\n"
        f"Backend retrieved candidates: {len(retrieved)}\n"
        f"Displayed/output rows: {len(demo_output)}"
    )

    return status, demo_output.head(20), str(output_path)


demo = gr.Interface(
    fn=rank_candidates_ui,
    inputs=[
        gr.Textbox(
            label="Job Description",
            lines=10,
            placeholder="Paste a job description here. If blank, the app uses data/raw/job_description.docx when available.",
        ),
        gr.File(label="Small Candidate Sample JSON/JSONL/CSV", file_types=[".json", ".jsonl", ".csv"]),
        gr.Slider(10, 100, value=100, step=10, label="Number of Output Candidates"),
    ],
    outputs=[
        gr.Textbox(label="Run Status"),
        gr.Dataframe(label="Top Ranked Candidates"),
        gr.File(label="Download Ranked CSV"),
    ],
    title="RecruiterFit AI",
    description="Upload a small candidate sample or use the bundled sample to run the ranking pipeline end to end.",
)


if __name__ == "__main__":
    demo.launch(share=True)
