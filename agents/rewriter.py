"""
Rewriter Agent — Uses Gemini to craft an optimized LinkedIn profile.
Incorporates market research and analysis weaknesses to maximize recruiter impact.
"""

import json
import re
import google.generativeai as genai


def rewrite_profile(
    target_job: str,
    headline: str,
    about: str,
    skills: str,
    experience: str,
    analysis: dict,
    research: dict,
    gemini_api_key: str,
) -> dict:
    """
    Rewrite the user's LinkedIn profile sections for maximum impact.

    Args:
        target_job:     The role the user is targeting.
        headline:       Current LinkedIn headline.
        about:          Current About / Summary section.
        skills:         Comma-separated list of current skills.
        experience:     Key experience bullet points (optional).
        analysis:       Dict returned by analyze_profile.
        research:       Dict returned by research_job_role.
        gemini_api_key: A valid Google Gemini API key.

    Returns:
        A dict with keys:
            headline         – str        Single best optimized headline.
            headline_options – list[str]  2–3 alternative headline variants.
            about            – str        Full optimized About section.
            skills           – list[str]  Top 15 skills to list on LinkedIn.
            featured_keywords– list[str]  High-value ATS keywords to embed.
            recruiter_tip    – str        One actionable tip for the recruiter view.
    """
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # ── Build context strings ─────────────────────────────────────────────────
    if isinstance(research, dict):
        top_skills_str   = ", ".join(research.get("top_skills", [])[:12])
        trends_str       = "; ".join(research.get("trends", [])[:4])
        certs_str        = ", ".join(research.get("certifications", [])[:5])
        market_str       = research.get("market_demand", "")
    else:
        top_skills_str = trends_str = certs_str = market_str = str(research)

    weaknesses_str   = "; ".join(analysis.get("weaknesses", []))
    missing_kw_str   = ", ".join(analysis.get("missing_keywords", []))
    missing_sk_str   = ", ".join(analysis.get("missing_skills", []))
    improvements_str = "; ".join(analysis.get("priority_improvements", []))

    prompt = f"""You are a world-class LinkedIn profile writer and personal branding expert.

Your task is to rewrite a LinkedIn profile for someone targeting **{target_job}**.

## Current Profile
**Headline:** {headline or "(not provided)"}
**About:** {about or "(not provided)"}
**Skills:** {skills or "(not provided)"}
**Experience:** {experience or "(not provided)"}

## Analysis Findings
- Weaknesses to fix: {weaknesses_str or "N/A"}
- Missing keywords: {missing_kw_str or "N/A"}
- Missing skills: {missing_sk_str or "N/A"}
- Priority improvements: {improvements_str or "N/A"}

## Market Research
- Top in-demand skills: {top_skills_str or "N/A"}
- Industry trends: {trends_str or "N/A"}
- Recommended certifications: {certs_str or "N/A"}
- Market context: {market_str or "N/A"}

## Writing Guidelines
1. **Headline** (max 220 chars): Lead with the job title, add 2–3 key skills/tools, end with a value proposition. Use | as separator. Be specific and keyword-rich.
2. **About** (1500–2000 chars): Start with a compelling hook. Use first person. Include quantified achievements where possible. Weave in missing keywords naturally. End with a clear call to action. Use short paragraphs for readability.
3. **Skills**: Select the 15 most impactful skills that match the target role and market demand.
4. **Featured keywords**: 8–12 high-value ATS keywords that should appear throughout the profile.
5. **Recruiter tip**: One specific, actionable tip to make the profile stand out to recruiters.

Respond with ONLY a valid JSON object — no markdown fences, no extra text — using exactly this structure:
{{
  "headline": "<single best headline>",
  "headline_options": [
    "<alternative headline 1>",
    "<alternative headline 2>",
    "<alternative headline 3>"
  ],
  "about": "<full optimized about section — use \\n for paragraph breaks>",
  "skills": ["<skill 1>", "<skill 2>", ..., "<skill 15>"],
  "featured_keywords": ["<keyword 1>", "<keyword 2>", ..., "<keyword 10>"],
  "recruiter_tip": "<one actionable recruiter tip>"
}}
"""

    response = model.generate_content(prompt)
    raw = response.text.strip()

    return _parse_json_response(raw, _default_rewrite(target_job, headline, about, skills))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str, fallback: dict) -> dict:
    """Extract and parse the first JSON object found in the model response."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

    try:
        data = json.loads(raw)
        return _coerce_rewrite(data, fallback)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]+\}", raw)
    if match:
        try:
            data = json.loads(match.group(0))
            return _coerce_rewrite(data, fallback)
        except json.JSONDecodeError:
            pass

    return fallback


def _coerce_rewrite(data: dict, fallback: dict) -> dict:
    """Ensure all expected keys exist and values have the correct types."""
    str_keys  = ["headline", "about", "recruiter_tip"]
    list_keys = ["headline_options", "skills", "featured_keywords"]
    result = {}
    for k in str_keys:
        val = data.get(k, fallback[k])
        result[k] = val if isinstance(val, str) else fallback[k]
    for k in list_keys:
        val = data.get(k, fallback[k])
        result[k] = val if isinstance(val, list) else fallback[k]
    return result


def _default_rewrite(target_job: str, headline: str, about: str, skills: str) -> dict:
    skills_list = [s.strip() for s in (skills or "").split(",") if s.strip()][:15]
    return {
        "headline": headline or f"{target_job} | Open to Opportunities",
        "headline_options": [
            f"{target_job} | Driving Impact Through Technology",
            f"Experienced {target_job} | Results-Oriented Professional",
        ],
        "about": about or (
            f"Passionate {target_job} with a track record of delivering high-quality solutions. "
            "Committed to continuous learning and making a measurable impact. "
            "Open to exciting new opportunities — let's connect!"
        ),
        "skills": skills_list or ["Communication", "Problem Solving", "Teamwork"],
        "featured_keywords": [target_job],
        "recruiter_tip": (
            "Add quantified achievements (e.g. 'reduced latency by 30%') to every "
            "experience entry — numbers make your profile 40% more likely to get a response."
        ),
    }
