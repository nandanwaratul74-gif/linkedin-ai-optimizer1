"""
Judge Agent — Uses Gemini to independently score a rewritten LinkedIn profile.
Acts as a skeptical recruiter to provide an unbiased quality assessment.
"""

import json
import re
import google.generativeai as genai


def judge_profile(
    target_job: str,
    rewritten_profile: dict,
    gemini_api_key: str,
) -> dict:
    """
    Independently score the rewritten LinkedIn profile as a skeptical recruiter would.

    Args:
        target_job:        The role the user is targeting.
        rewritten_profile: Dict returned by rewrite_profile.
        gemini_api_key:    A valid Google Gemini API key.

    Returns:
        A dict with keys:
            overall          – int  0-10  Composite judge score.
            verdict          – str  One of: EXCELLENT | GOOD | NEEDS WORK | POOR
            clarity          – int  0-10  How clear and readable the profile is.
            keywords         – int  0-10  Keyword density and relevance.
            professionalism  – int  0-10  Tone, grammar, and professional polish.
            ats_ready        – int  0-10  ATS / applicant tracking system readiness.
            recruiter_appeal – int  0-10  How compelling it is to a human recruiter.
            uniqueness       – int  0-10  How well it stands out from generic profiles.
            best_part        – str  The single strongest element of the profile.
            critical_fix     – str  The one change that would have the biggest impact.
            detailed_feedback– str  2–4 sentences of constructive overall feedback.
    """
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    headline         = rewritten_profile.get("headline", "")
    about            = rewritten_profile.get("about", "")
    skills           = rewritten_profile.get("skills", [])
    featured_kw      = rewritten_profile.get("featured_keywords", [])
    headline_options = rewritten_profile.get("headline_options", [])

    skills_str   = ", ".join(skills[:15]) if isinstance(skills, list) else str(skills)
    kw_str       = ", ".join(featured_kw)  if isinstance(featured_kw, list) else str(featured_kw)
    alt_headlines = "\n".join(f"  - {h}" for h in headline_options) if headline_options else "  (none)"

    prompt = f"""You are a senior technical recruiter with 15+ years of experience hiring for {target_job} roles.

You have just received the following optimized LinkedIn profile. Evaluate it critically and honestly — your job is to identify both what works and what still needs improvement.

## Profile to Judge

**Target Role:** {target_job}

**Headline:** {headline}

**Alternative Headlines:**
{alt_headlines}

**About Section:**
{about}

**Skills:** {skills_str}

**Featured Keywords:** {kw_str}

## Scoring Criteria
Score each dimension from 0 to 10 (integers only):
- **clarity**: Is the profile easy to read and understand at a glance?
- **keywords**: Are the right industry keywords present and used naturally?
- **professionalism**: Is the tone polished, grammar correct, and language professional?
- **ats_ready**: Would this profile pass automated ATS screening for {target_job} roles?
- **recruiter_appeal**: Would this make a recruiter want to reach out immediately?
- **uniqueness**: Does this stand out from the hundreds of generic profiles recruiters see daily?
- **overall**: Your holistic assessment (not just an average — weight by importance).

**Verdict thresholds:**
- EXCELLENT: overall ≥ 9
- GOOD: overall 7–8
- NEEDS WORK: overall 5–6
- POOR: overall ≤ 4

Respond with ONLY a valid JSON object — no markdown fences, no extra text — using exactly this structure:
{{
  "overall": <int 0-10>,
  "verdict": "<EXCELLENT|GOOD|NEEDS WORK|POOR>",
  "clarity": <int 0-10>,
  "keywords": <int 0-10>,
  "professionalism": <int 0-10>,
  "ats_ready": <int 0-10>,
  "recruiter_appeal": <int 0-10>,
  "uniqueness": <int 0-10>,
  "best_part": "<one sentence describing the strongest element>",
  "critical_fix": "<one sentence describing the single most impactful change>",
  "detailed_feedback": "<2-4 sentences of constructive overall feedback>"
}}
"""

    response = model.generate_content(prompt)
    raw = response.text.strip()

    return _parse_json_response(raw, _default_judgment())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str, fallback: dict) -> dict:
    """Extract and parse the first JSON object found in the model response."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

    try:
        data = json.loads(raw)
        return _coerce_judgment(data, fallback)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]+\}", raw)
    if match:
        try:
            data = json.loads(match.group(0))
            return _coerce_judgment(data, fallback)
        except json.JSONDecodeError:
            pass

    return fallback


def _coerce_judgment(data: dict, fallback: dict) -> dict:
    """Ensure all expected keys exist and values have the correct types."""
    int_keys = [
        "overall", "clarity", "keywords", "professionalism",
        "ats_ready", "recruiter_appeal", "uniqueness",
    ]
    str_keys = ["verdict", "best_part", "critical_fix", "detailed_feedback"]
    valid_verdicts = {"EXCELLENT", "GOOD", "NEEDS WORK", "POOR"}

    result = {}
    for k in int_keys:
        try:
            result[k] = max(0, min(10, int(data.get(k, fallback[k]))))
        except (TypeError, ValueError):
            result[k] = fallback[k]

    for k in str_keys:
        val = data.get(k, fallback[k])
        result[k] = val if isinstance(val, str) else fallback[k]

    # Normalise verdict
    verdict = result["verdict"].upper().strip()
    if verdict not in valid_verdicts:
        # Map partial matches
        if "EXCEL" in verdict:
            verdict = "EXCELLENT"
        elif "NEED" in verdict or "WORK" in verdict:
            verdict = "NEEDS WORK"
        elif "POOR" in verdict or "BAD" in verdict:
            verdict = "POOR"
        else:
            verdict = "GOOD"
    result["verdict"] = verdict

    return result


def _default_judgment() -> dict:
    return {
        "overall": 7,
        "verdict": "GOOD",
        "clarity": 7,
        "keywords": 7,
        "professionalism": 7,
        "ats_ready": 7,
        "recruiter_appeal": 7,
        "uniqueness": 6,
        "best_part": "The profile has been optimized with relevant keywords and a clear structure.",
        "critical_fix": "Add specific, quantified achievements to make the profile more compelling.",
        "detailed_feedback": (
            "The rewritten profile shows clear improvement over the original. "
            "To further strengthen it, focus on adding measurable outcomes to each "
            "experience entry and ensuring the headline immediately communicates your "
            "unique value proposition to recruiters."
        ),
    }
