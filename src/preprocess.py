import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer


ID_KEYS = ["candidate_id", "id", "user_id", "profile_id"]
NAME_KEYS = ["candidate_name", "name", "full_name"]


def load_candidates(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ["candidates", "data", "profiles", "users"]:
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
        return pd.DataFrame(data)

    if suffix == "":
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return pd.DataFrame()
        if text[0] == "[" or text[0] == "{":
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    for key in ["candidates", "data", "profiles", "users"]:
                        if key in data and isinstance(data[key], list):
                            data = data[key]
                            break
                return pd.DataFrame(data)
            except json.JSONDecodeError:
                pass
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return pd.DataFrame(rows)

    raise ValueError(f"Unsupported candidate file format: {path.suffix}")


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return re.sub(r"\s+", " ", str(value)).strip()


def first_present(row: pd.Series, keys: list[str], default: str = "") -> str:
    for key in keys:
        if key in row and clean_text(row[key]):
            return clean_text(row[key])
    return default


def get_nested(row: pd.Series, key: str, default: Any = "") -> Any:
    if "." not in key:
        return row.get(key, default)
    value: Any = row
    for part in key.split("."):
        if isinstance(value, pd.Series):
            value = value.get(part, default)
        elif isinstance(value, dict):
            value = value.get(part, default)
        else:
            return default
    return value


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def format_skills(skills: Any) -> tuple[str, float]:
    skills = parse_jsonish(skills)
    if not skills:
        return "", 0.0

    proficiency_weight = {"beginner": 0.4, "intermediate": 0.7, "advanced": 1.0, "expert": 1.0}
    lines: list[str] = []
    scores: list[float] = []

    if isinstance(skills, list):
        for item in skills:
            if isinstance(item, dict):
                name = clean_text(item.get("name", ""))
                proficiency = clean_text(item.get("proficiency", "")).lower()
                endorsements = float(item.get("endorsements") or 0)
                duration = float(item.get("duration_months") or 0)
                if not name:
                    continue
                lines.append(
                    f"{name} - {proficiency or 'unknown'}, {int(duration)} months, {int(endorsements)} endorsements"
                )
                p_score = proficiency_weight.get(proficiency, 0.5)
                d_score = min(duration / 48.0, 1.0)
                e_score = min(endorsements / 20.0, 1.0)
                scores.append((0.5 * p_score) + (0.3 * d_score) + (0.2 * e_score))
            else:
                text = clean_text(item)
                if text:
                    lines.append(text)
                    scores.append(0.4)
    else:
        text = clean_text(skills)
        return text, 0.4 if text else 0.0

    return "; ".join(lines), round(sum(scores) / max(len(scores), 1), 4)


def days_since(date_value: Any) -> int | None:
    text = clean_text(date_value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text).date()
    except ValueError:
        return None
    return max((date.today() - parsed).days, 0)


def score_redrob_signals(signals: Any) -> tuple[str, float]:
    signals = parse_jsonish(signals)
    if not isinstance(signals, dict):
        return "", 0.0

    profile_complete = float(signals.get("profile_completeness_score") or 0) / 100
    response_rate = float(signals.get("recruiter_response_rate") or 0)
    interview_rate = float(signals.get("interview_completion_rate") or 0)
    offer_rate = signals.get("offer_acceptance_rate")
    offer_rate = 0.5 if offer_rate in [None, -1] else float(offer_rate)
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.0

    active_days = days_since(signals.get("last_active_date"))
    recency_score = 0.0
    if active_days is not None:
        if active_days <= 7:
            recency_score = 1.0
        elif active_days <= 30:
            recency_score = 0.8
        elif active_days <= 90:
            recency_score = 0.45
        else:
            recency_score = 0.1

    response_hours = float(signals.get("avg_response_time_hours") or 999)
    response_speed = 1.0 if response_hours <= 12 else 0.8 if response_hours <= 24 else 0.5 if response_hours <= 72 else 0.2

    recruiter_interest = min(float(signals.get("saved_by_recruiters_30d") or 0) / 10, 1.0)
    search_interest = min(float(signals.get("search_appearance_30d") or 0) / 100, 1.0)
    applications = min(float(signals.get("applications_submitted_30d") or 0) / 20, 1.0)
    verified = (
        float(bool(signals.get("verified_email")))
        + float(bool(signals.get("verified_phone")))
        + float(bool(signals.get("linkedin_connected")))
    ) / 3

    github = float(signals.get("github_activity_score") or 0)
    github = 0.0 if github < 0 else min(github / 100, 1.0)

    score = (
        0.15 * profile_complete
        + 0.15 * recency_score
        + 0.15 * response_rate
        + 0.10 * response_speed
        + 0.10 * open_to_work
        + 0.10 * interview_rate
        + 0.05 * offer_rate
        + 0.07 * recruiter_interest
        + 0.05 * search_interest
        + 0.03 * applications
        + 0.03 * verified
        + 0.02 * github
    )

    summary = (
        f"profile completeness {profile_complete * 100:.0f}%, "
        f"last active {active_days if active_days is not None else 'unknown'} days ago, "
        f"open to work {bool(signals.get('open_to_work_flag'))}, "
        f"recruiter response rate {response_rate:.0%}, "
        f"avg response time {response_hours:.0f} hours, "
        f"interview completion {interview_rate:.0%}, "
        f"saved by recruiters 30d {int(signals.get('saved_by_recruiters_30d') or 0)}"
    )
    return summary, round(score, 4)


def build_profile_text(row: pd.Series) -> tuple[str, float]:
    skills_text, skill_score = format_skills(row.get("skills", ""))
    redrob_summary, activity_score = score_redrob_signals(row.get("redrob_signals", ""))

    parts = [
        f"Candidate Name: {first_present(row, NAME_KEYS, clean_text(get_nested(row, 'profile.anonymized_name', 'Unknown')))}",
        f"Headline: {clean_text(row.get('headline', get_nested(row, 'profile.headline', '')))}",
        f"Current Role: {clean_text(row.get('current_role', row.get('title', get_nested(row, 'profile.current_title', ''))))}",
        f"Experience Years: {clean_text(row.get('experience_years', row.get('years_experience', get_nested(row, 'profile.years_of_experience', ''))))}",
        f"Current Industry: {clean_text(get_nested(row, 'profile.current_industry', ''))}",
        f"Location: {clean_text(get_nested(row, 'profile.location', ''))}, {clean_text(get_nested(row, 'profile.country', ''))}",
        f"Skills: {skills_text}",
        f"Work History: {clean_text(row.get('work_history', row.get('experience', row.get('career_history', ''))))}",
        f"Projects: {clean_text(row.get('projects', ''))}",
        f"Education: {clean_text(row.get('education', ''))}",
        f"Certifications: {clean_text(row.get('certifications', ''))}",
        f"Redrob Signals: {redrob_summary or clean_text(row.get('platform_activity', row.get('activity', '')))}",
        f"Achievements: {clean_text(row.get('achievements', ''))}",
        f"Summary: {clean_text(row.get('summary', row.get('bio', get_nested(row, 'profile.summary', ''))))}",
    ]
    return "\n".join(part for part in parts if not part.endswith(": ")), skill_score, activity_score


def normalize_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    records = []

    for pos, (_, row) in enumerate(df.iterrows()):
        profile_text, skill_score, activity_score = build_profile_text(row)
        candidate_id = first_present(row, ID_KEYS, str(pos))
        candidate_name = first_present(row, NAME_KEYS, clean_text(get_nested(row, "profile.anonymized_name", "Unknown")))
        records.append(
            {
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "profile_text": profile_text,
                "structured_score": skill_score,
                "activity_score": activity_score,
            }
        )

    out = pd.DataFrame(records)
    out = out.drop_duplicates(subset=["candidate_id"], keep="first")
    out = out[out["profile_text"].str.len() > 20].reset_index(drop=True)
    return out


def prefilter_raw_active_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap raw-schema filter to remove clearly inactive candidates before embedding."""
    if "redrob_signals" not in df.columns:
        return df.copy()

    keep_mask = []
    for _, row in df.iterrows():
        signals = parse_jsonish(row.get("redrob_signals", {}))
        if not isinstance(signals, dict):
            keep_mask.append(True)
            continue

        active_days = days_since(signals.get("last_active_date"))
        is_recent = active_days is not None and active_days <= 60
        is_moderately_recent = active_days is not None and active_days <= 180
        is_open = bool(signals.get("open_to_work_flag"))
        response_rate = float(signals.get("recruiter_response_rate") or 0)
        saved_by_recruiters = int(signals.get("saved_by_recruiters_30d") or 0)
        applications = int(signals.get("applications_submitted_30d") or 0)

        keep_mask.append(
            is_recent
            or response_rate >= 0.50
            or saved_by_recruiters >= 3
            or (is_open and is_moderately_recent)
            or (is_open and response_rate >= 0.20)
            or (is_open and saved_by_recruiters > 0)
            or (is_open and applications > 0)
        )

    return df[pd.Series(keep_mask, index=df.index)].reset_index(drop=True)


def prefilter_raw_role_candidates(
    df: pd.DataFrame,
    job_description: str,
    top_n: int = 10000,
) -> pd.DataFrame:
    """Fast job-specific lexical prefilter before expensive normalization."""
    if top_n <= 0 or len(df) <= top_n:
        out = df.copy()
        out["raw_role_score"] = 1.0
        return out.reset_index(drop=True)

    texts = [build_raw_prefilter_text(row) for _, row in df.iterrows()]
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
    keep = scores.argsort()[-top_n:][::-1]

    out = df.iloc[keep].copy().reset_index(drop=True)
    out["raw_role_score"] = scores[keep]
    return out


def build_raw_prefilter_text(row: pd.Series) -> str:
    profile = parse_jsonish(row.get("profile", {}))
    skills = parse_jsonish(row.get("skills", []))
    career = parse_jsonish(row.get("career_history", []))
    education = parse_jsonish(row.get("education", []))

    parts: list[str] = []
    if isinstance(profile, dict):
        for key in ["headline", "summary", "current_title", "current_industry"]:
            parts.append(clean_text(profile.get(key, "")))

    if isinstance(skills, list):
        for skill in skills:
            if isinstance(skill, dict):
                parts.append(clean_text(skill.get("name", "")))
                parts.append(clean_text(skill.get("proficiency", "")))

    if isinstance(career, list):
        for item in career[:3]:
            if isinstance(item, dict):
                parts.append(clean_text(item.get("title", "")))
                parts.append(clean_text(item.get("industry", "")))
                parts.append(clean_text(item.get("description", "")))

    if isinstance(education, list):
        for item in education[:2]:
            if isinstance(item, dict):
                parts.append(clean_text(item.get("degree", "")))
                parts.append(clean_text(item.get("field_of_study", "")))

    return " ".join(part for part in parts if part)


def filter_active_candidates(candidates: pd.DataFrame, min_activity_score: float = 0.15) -> pd.DataFrame:
    """Remove clearly unavailable profiles while keeping borderline strong candidates."""
    if "activity_score" not in candidates.columns:
        return candidates.copy()

    filtered = candidates[candidates["activity_score"].astype(float) >= min_activity_score].copy()
    return filtered.reset_index(drop=True)
