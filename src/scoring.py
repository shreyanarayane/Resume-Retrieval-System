import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def minmax(series: pd.Series) -> pd.Series:
    low = series.min()
    high = series.max()
    if high == low:
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - low) / (high - low)


def rank_without_llm(candidates: pd.DataFrame) -> pd.DataFrame:
    ranked = candidates.copy()
    ranked["semantic_norm"] = minmax(ranked["semantic_score"].astype(float))
    ranked["structured_norm"] = ranked["structured_score"].astype(float).clip(0, 1)
    if "activity_score" not in ranked.columns:
        ranked["activity_score"] = 0.0
    ranked["activity_norm"] = ranked["activity_score"].astype(float).clip(0, 1)
    ranked["final_score"] = (
        0.55 * ranked["semantic_norm"] + 0.35 * ranked["structured_norm"] + 0.10 * ranked["activity_norm"]
    ) * 100
    ranked["reasoning"] = ranked.apply(build_reasoning, axis=1)
    return sort_ranked(ranked)


def rank_candidates(candidates: pd.DataFrame, job_description: str) -> pd.DataFrame:
    ranked = rank_without_llm(candidates)
    requirements = extract_job_requirements(job_description)
    jd_signals = ranked.apply(lambda row: compute_jd_specific_signals(row, requirements), axis=1, result_type="expand")
    ranked = pd.concat([ranked, jd_signals], axis=1)

    ranked["final_score"] = (
        ranked["final_score"]
        * (0.45 + 0.55 * ranked["jd_evidence_score"])
        * (0.35 + 0.65 * ranked["primary_jd_evidence_score"])
        * (0.30 + 0.70 * ranked["core_experience_score"])
        * ranked["experience_penalty"]
    ).clip(lower=0)
    ranked["reasoning"] = ranked.apply(lambda row: build_reasoning(row, requirements), axis=1)
    return sort_ranked(ranked)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = sort_ranked(df)
    columns = [
        "candidate_id",
        "rank",
        "score",
        "reasoning",
    ]
    out = out.rename(columns={"final_score": "score"})
    out["score"] = (out["score"] / 100).round(6)
    return out[columns]


def save_submission(df: pd.DataFrame, path: str, top_n: int = 100) -> pd.DataFrame:
    submission = finalize(df).head(top_n)
    if len(submission) != top_n:
        raise ValueError(f"Submission must contain {top_n} rows, got {len(submission)}.")
    if submission["rank"].tolist() != list(range(1, top_n + 1)):
        raise ValueError("Submission ranks must be exactly 1 through 100.")
    if submission["candidate_id"].duplicated().any():
        raise ValueError("Submission contains duplicate candidate_id values.")
    if not submission["score"].is_monotonic_decreasing:
        raise ValueError("Submission scores must be monotonically non-increasing.")

    from pathlib import Path

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def sort_ranked(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    if "rank" in out.columns:
        out = out.drop(columns=["rank"])
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def build_reasoning(row: pd.Series, requirements: dict | None = None) -> str:
    profile_text = str(row.get("profile_text", ""))
    title = extract_field(profile_text, "Current Role") or "Candidate"
    years = extract_field(profile_text, "Experience Years")
    skills = extract_top_skills(profile_text)
    redrob = extract_field(profile_text, "Redrob Signals")

    matched_terms = row.get("matched_primary_jd_terms", row.get("matched_jd_terms", ""))
    matched = [term for term in str(matched_terms).split("; ") if term and term not in GENERIC_JD_WORDS]
    matched_skills = [skill for skill in skills if any(term in skill.lower() or skill.lower() in term for term in matched)]
    core_terms = [term for term in str(row.get("matched_core_jd_terms", "")).split("; ") if term]
    evidence_terms = matched_skills or [term for term in matched if len(term) > 3 and term not in GENERIC_JD_WORDS]
    evidence_terms = core_terms[:2] + [term for term in evidence_terms if term not in core_terms]
    skill_phrase = ", ".join(evidence_terms[:3]) if evidence_terms else ", ".join(skills[:3]) if skills else "relevant listed skills"
    years_phrase = f" with {years} years of experience" if years else ""
    fit_phrase = "strong fit" if float(row.get("final_score", 0)) >= 75 else "reasonable fit" if float(row.get("final_score", 0)) >= 55 else "borderline fit"

    signal_phrase = summarize_redrob(redrob, float(row.get("activity_score", 0)))
    concern = jd_concern_text(row, requirements)
    evidence_label = "strong work-history evidence" if float(row.get("core_experience_score", 1.0)) >= 0.25 else "listed skill evidence but limited work-history proof"
    return (
        f"{title}{years_phrase} and {evidence_label} in {skill_phrase}. "
        f"{signal_phrase}, making them a {fit_phrase} for the JD.{concern}"
    )


def extract_field(text: str, field_name: str) -> str:
    match = re.search(rf"^{re.escape(field_name)}:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def extract_top_skills(text: str, limit: int = 3) -> list[str]:
    skills_line = extract_field(text, "Skills")
    if not skills_line:
        return []

    parsed: list[tuple[str, str, int, int]] = []
    for item in skills_line.split(";"):
        item = item.strip()
        match = re.match(r"(.+?)\s+-\s+(\w+),\s+(\d+)\s+months,\s+(\d+)\s+endorsements", item)
        if not match:
            continue
        name, proficiency, months, endorsements = match.groups()
        parsed.append((name.strip(), proficiency.lower(), int(months), int(endorsements)))

    proficiency_weight = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    parsed.sort(key=lambda x: (proficiency_weight.get(x[1], 0), x[2], x[3]), reverse=True)
    return [name for name, _, _, _ in parsed[:limit]]


def summarize_redrob(redrob: str, activity_score: float) -> str:
    if not redrob:
        return "Redrob signals are limited but included in the ranking"

    open_to_work = "open to work True" in redrob
    saved_match = re.search(r"saved by recruiters 30d\s+(\d+)", redrob)
    active_match = re.search(r"last active\s+(\d+)\s+days ago", redrob)
    response_match = re.search(r"recruiter response rate\s+(\d+)%", redrob)

    details = []
    if open_to_work:
        details.append("open-to-work status")
    if active_match:
        days = int(active_match.group(1))
        if days <= 30:
            details.append("recent activity")
        elif days <= 90:
            details.append("moderate recent activity")
        else:
            details.append("some recency concern")
    if response_match:
        response = int(response_match.group(1))
        if response >= 50:
            details.append("healthy recruiter response rate")
        elif response < 25:
            details.append("lower recruiter response rate")
    if saved_match and int(saved_match.group(1)) > 0:
        details.append("recruiter saves")

    if not details:
        details.append("behavioral availability evidence")

    prefix = "Redrob signals show" if activity_score >= 0.35 else "Redrob signals add some concern but show"
    return f"{prefix} {', '.join(details[:3])}"


def extract_job_requirements(job_description: str) -> dict:
    text = job_description.lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", text)
    stop_words = set(ENGLISH_STOP_WORDS) | GENERIC_JD_WORDS

    phrases: dict[str, int] = {}
    for n in [1, 2, 3]:
        for i in range(len(tokens) - n + 1):
            phrase_tokens = tokens[i : i + n]
            useful = [token for token in phrase_tokens if token not in stop_words and len(token) > 2]
            if not useful:
                continue
            phrase = " ".join(phrase_tokens)
            if phrase in GENERIC_JD_WORDS:
                continue
            phrases[phrase] = phrases.get(phrase, 0) + 1

    terms = sorted(phrases, key=lambda term: (phrases[term], len(term.split()), len(term)), reverse=True)
    terms = terms[:30]

    years_matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)", text)
    min_years = min([float(match) for match in years_matches], default=None)

    return {"terms": terms, "min_years": min_years}


def compute_jd_specific_signals(row: pd.Series, requirements: dict) -> pd.Series:
    text = str(row.get("profile_text", "")).lower()
    profile_text = str(row.get("profile_text", ""))
    years = parse_years(extract_field(str(row.get("profile_text", "")), "Experience Years"))
    skills = [skill.lower() for skill in extract_top_skills(profile_text, limit=20)]
    primary_text = " ".join(
        [
            extract_field(profile_text, "Current Role"),
            extract_field(profile_text, "Headline"),
            extract_field(profile_text, "Skills"),
        ]
    ).lower()
    work_text = extract_work_history_text(profile_text).lower()

    terms = requirements.get("terms", [])
    matched_terms = []
    matched_primary_terms = []
    matched_core_terms = []
    evidence = 0.0
    primary_evidence = 0.0
    core_evidence = 0.0
    for term in terms:
        term_l = term.lower()
        if any(term_l in skill or skill in term_l for skill in skills):
            evidence += 1.5
            primary_evidence += 1.5
            matched_terms.append(term)
            matched_primary_terms.append(term)
        elif term_l in primary_text:
            evidence += 1.2
            primary_evidence += 1.2
            matched_terms.append(term)
            matched_primary_terms.append(term)
        elif term_l in text:
            evidence += 1.0
            matched_terms.append(term)

        if term_l in work_text and has_implementation_context(work_text, term_l):
            core_evidence += 1.5
            matched_core_terms.append(term)

    jd_evidence_score = min(evidence / max(len(terms[:15]), 1), 1.0)
    primary_jd_evidence_score = min(primary_evidence / max(len(terms[:12]), 1), 1.0)
    core_experience_score = min(core_evidence / max(len(terms[:10]), 1), 1.0)
    min_years = requirements.get("min_years")
    exp_penalty = 1.0
    if min_years is not None:
        if years == 0:
            exp_penalty = 0.85
        elif years < max(min_years - 1, 0):
            exp_penalty = 0.65
        elif years < min_years:
            exp_penalty = 0.85

    return pd.Series(
        {
            "jd_evidence_score": jd_evidence_score,
            "primary_jd_evidence_score": primary_jd_evidence_score,
            "core_experience_score": core_experience_score,
            "experience_penalty": exp_penalty,
            "matched_jd_terms": "; ".join(dict.fromkeys(matched_terms[:6])),
            "matched_primary_jd_terms": "; ".join(dict.fromkeys(matched_primary_terms[:6])),
            "matched_core_jd_terms": "; ".join(dict.fromkeys(matched_core_terms[:6])),
            "years_experience_num": years,
        }
    )


def parse_years(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def jd_concern_text(row: pd.Series, requirements: dict | None = None) -> str:
    concerns = []
    if requirements and float(row.get("jd_evidence_score", 1.0)) < 0.35:
        concerns.append("limited direct evidence for the JD's extracted requirements")
    if requirements and float(row.get("primary_jd_evidence_score", 1.0)) < 0.20:
        concerns.append("JD evidence is mostly outside core title/skills")
    if requirements and float(row.get("core_experience_score", 1.0)) < 0.20:
        concerns.append("limited work-history evidence of applying the JD requirements")
    min_years = requirements.get("min_years") if requirements else None
    years = float(row.get("years_experience_num", 0) or 0)
    if min_years is not None and years and years < min_years:
        concerns.append("experience appears below the JD's preferred seniority band")
    return f" Concern: {', '.join(concerns)}." if concerns else ""


GENERIC_JD_WORDS = {
    "candidate",
    "candidates",
    "company",
    "companies",
    "experience",
    "experienced",
    "role",
    "roles",
    "responsibility",
    "responsibilities",
    "requirement",
    "requirements",
    "required",
    "preferred",
    "strong",
    "good",
    "excellent",
    "work",
    "working",
    "team",
    "teams",
    "build",
    "building",
    "develop",
    "developing",
    "years",
    "month",
    "months",
    "redrob",
    "we",
    "re",
    "going",
    "actually",
    "maybe",
    "candidate profile",
    "profile",
    "platform",
    "signal",
    "signals",
    "data",
    "dataset",
    "recruiter",
    "recruiters",
    "skills",
    "skill",
    "systems",
    "system",
    "product",
    "products",
    "engineering",
    "engineer",
    "engineers",
    "senior",
    "junior",
    "ability",
    "knowledge",
    "understanding",
}


IMPLEMENTATION_TERMS = {
    "built",
    "build",
    "owned",
    "designed",
    "implemented",
    "developed",
    "deployed",
    "shipped",
    "pipeline",
    "service",
    "inference",
    "trained",
    "fine-tuned",
    "integrated",
}


def extract_work_history_text(profile_text: str) -> str:
    match = re.search(r"Work History:\s*(.+?)\nEducation:", profile_text, flags=re.DOTALL)
    return match.group(1) if match else profile_text


def has_implementation_context(work_text: str, term: str) -> bool:
    sentences = re.split(r"[.!?]\s+|\\n", work_text)
    for sentence in sentences:
        has_term = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", sentence) is not None
        has_impl = any(re.search(rf"(?<!\w){re.escape(impl)}(?!\w)", sentence) for impl in IMPLEMENTATION_TERMS)
        if has_term and has_impl:
            return True
    return False
