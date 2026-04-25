"""
Analyzer Agent — Uses Gemini to score a LinkedIn profile against a target job role.
Produces a structured dict consumed by the Streamlit UI and the Rewriter agent.
"""

import json
import re
import google.generativeai as genai


def analyze_profile(
    target_job: str,
    headline: str,
    about: str,
    skills: str,
    experience: str,
    research: dict,
    gemini_api_key: str,
) -> dict:
    """
    Analyze the user's LinkedIn profile against the target job using Gemini.

    Args:
        target_job:     The role the user is targeting.
        headline:       Current LinkedIn headline.
        about:          Current About / Summary section.
        skills:         Comma-separated list of current skills.
        experience:     Key experience bullet points (optional).
        research:       Dict returned by research_job_role (may be a string on error).
        gemini_api_key: A valid Google Gemini API key.

    Returns:
        A dict with keys:
            overall_score         – int  0-10
            headline_score        – int  0-10
            about_score           – int  0-10
            keyword_score         – int  0-10
            skills_score          – int  0-10
            ats_compatibility     – int  0-10
            strengths             – list[str]
            weaknesses            – list[str]
            missing_keywords      – list[str]
            priority_improvements – list[str]
            missing_skills        – list[str]
            missing_certifications– list[str]
    """
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Safely stringify research context
    if isinstance(research, dict):
        research_context = (
            f"Market trends: {', '.join(research.get('trends', [])[:3])}\n"
            f"Top skills in demand: {', '.join(research.get('top_skills', [])[:10])}\n"
            f"Recommended certifications: {', '.join(research.get('certifications', [])[:5])}\n"
            f"Market demand: {research.get('market_demand', 'N/A')}"
        )
    else:
        research_context = str(research)

    prompt = f"""You are an expert LinkedIn profile coach and ATS specialist.

Analyze the following LinkedIn profile for someone targeting the role of **{target_job}**.

## Current Profile
**Headline:** {headline or "(not provided)"}
**About:** {about or "(not provided)"}
**Skills:** {skills or "(not provided)"}
**Experience:** {experience or "(not provided)"}

## Market Research Context
{research_context}

## Task
Score the profile across 6 dimensions (each 0–10, integers only) and provide actionable feedback.

Respond with ONLY a valid JSON object — no markdown fences, no extra text — using exactly this structure:
{{
  "overall_score": <int 0-10>,
  "headline_score": <int 0-10>,
  "about_score": <int 0-10>,
  "keyword_score": <int 0-10>,
  "skills_score": <int 0-10>,
  "ats_compatibility": <int 0-10>,
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>", "<weakness 3>"],
  "missing_keywords": ["<keyword 1>", "<keyword 2>", "<keyword 3>", "<keyword 4>", "<keyword 5>"],
  "priority_improvements": ["<improvement 1>", "<improvement 2>", "<improvement 3>"],
  "missing_skills": ["<skill 1>", "<skill 2>", "<skill 3>", "<skill 4>"],
  "missing_certifications": ["<cert 1>", "<cert 2>", "<cert 3>"]
}}

Scoring guidelines:
- Be honest and critical — most profiles score 4–7 before optimization.
- overall_score should reflect the weighted average of the other scores.
- ats_compatibility measures how well the profile would pass automated resume screening.
- missing_keywords should be specific terms recruiters search for in this role.
- priority_improvements should be concrete, actionable steps ordered by impact.
"""

    response = model.generate_content(prompt)
    raw = response.text.strip()

    return _parse_json_response(raw, _default_analysis())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str, fallback: dict) -> dict:
    """Extract and parse the first JSON object found in the model response."""
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

    # Try direct parse first
    try:
        data = json.loads(raw)
        return _coerce_analysis(data, fallback)
    except json.JSONDecodeError:
        pass

    # Try to extract the first {...} block
    match = re.search(r"\{[\s\S]+\}", raw)
    if match:
        try:
            data = json.loads(match.group(0))
            return _coerce_analysis(data, fallback)
        except json.JSONDecodeError:
            pass

    return fallback


def _coerce_analysis(data: dict, fallback: dict) -> dict:
    """Ensure all expected keys exist and values have the correct types."""
    int_keys = [
        "overall_score", "headline_score", "about_score",
        "keyword_score", "skills_score", "ats_compatibility",
    ]
    list_keys = [
        "strengths", "weaknesses", "missing_keywords",
        "priority_improvements", "missing_skills", "missing_certifications",
    ]
    result = {}
    for k in int_keys:
        try:
            result[k] = max(0, min(10, int(data.get(k, fallback[k]))))
        except (TypeError, ValueError):
            result[k] = fallback[k]
    for k in list_keys:
        val = data.get(k, fallback[k])
        result[k] = val if isinstance(val, list) else fallback[k]
    return result


def _default_analysis() -> dict:
    return {
        "overall_score": 5,
        "headline_score": 5,
        "about_score": 5,
        "keyword_score": 5,
        "skills_score": 5,
        "ats_compatibility": 5,
        "strengths": ["Profile submitted for analysis"],
        "weaknesses": ["Unable to complete full analysis — please retry"],
        "missing_keywords": [],
        "priority_improvements": ["Re-run the optimizer to get detailed recommendations"],
        "missing_skills": [],
        "missing_certifications": [],
    }
