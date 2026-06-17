import gradio as gr
from pathlib import Path

from src.io_utils import read_job_description
from src.preprocess import load_candidates, normalize_candidates, prefilter_raw_active_candidates, prefilter_raw_role_candidates
from src.retrieval import retrieve_top_k
from src.scoring import rank_candidates, save_submission

def rank_candidates_ui(job_text, top_k):
    raw_path = Path("data/raw/candidates.jsonl")

    raw = load_candidates(raw_path)
    raw = prefilter_raw_active_candidates(raw)
    raw = prefilter_raw_role_candidates(raw, job_text, top_n=3500)

    candidates = normalize_candidates(raw)
    retrieved = retrieve_top_k(candidates, job_text, top_k=int(top_k), tfidf_prefilter_k=None)
    ranked = rank_candidates(retrieved, job_text)

    output_path = "outputs/gradio_ranked_candidates.csv"
    save_submission(ranked, output_path, top_n=100)

    return ranked[["rank", "candidate_id", "final_score", "reasoning"]].head(20), output_path

demo = gr.Interface(
    fn=rank_candidates_ui,
    inputs=[
        gr.Textbox(label="Job Description", lines=10),
        gr.Slider(100, 1000, value=1000, step=100, label="Semantic Retrieval Top-K"),
    ],
    outputs=[
        gr.Dataframe(label="Top Ranked Candidates"),
        gr.File(label="Download Submission CSV"),
    ],
    title="RecruiterFit AI",
)

demo.launch()