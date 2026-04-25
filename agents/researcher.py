"""
Researcher Agent — Uses Tavily to search LinkedIn trends for a target job role.
Returns market intelligence that feeds into the Analyzer and Rewriter agents.
"""

import json
from tavily import TavilyClient


def research_job_role(job_title: str, tavily_api_key: str) -> dict:
    """
    Search for LinkedIn trends, in-demand skills, certifications, salary range,
    and market demand for the given job title using the Tavily search API.

    Args:
        job_title:      The target job role to research (e.g. "Senior ML Engineer").
        tavily_api_key: A valid Tavily API key.

    Returns:
        A dict with keys:
            trends        – list[str]  Current LinkedIn / hiring trends for the role.
            top_skills    – list[str]  Most in-demand technical and soft skills.
            certifications– list[str]  Recommended certifications for the role.
            salary_range  – str        Typical salary range (e.g. "$120k – $160k").
            market_demand – str        Brief description of hiring demand / outlook.
    """
    client = TavilyClient(api_key=tavily_api_key)

    # ── Query 1: skills & trends ──────────────────────────────────────────────
    skills_response = client.search(
        query=f"{job_title} LinkedIn profile top skills requirements 2024 2025",
        search_depth="advanced",
        max_results=5,
        include_answer=True,
    )

    # ── Query 2: certifications ───────────────────────────────────────────────
    cert_response = client.search(
        query=f"best certifications for {job_title} career 2024 2025",
        search_depth="basic",
        max_results=3,
        include_answer=True,
    )

    # ── Query 3: salary & demand ──────────────────────────────────────────────
    market_response = client.search(
        query=f"{job_title} salary range job market demand hiring trends 2024 2025",
        search_depth="basic",
        max_results=3,
        include_answer=True,
    )

    # ── Parse answers ─────────────────────────────────────────────────────────
    skills_answer  = skills_response.get("answer", "") or ""
    cert_answer    = cert_response.get("answer", "")   or ""
    market_answer  = market_response.get("answer", "") or ""

    # Build a combined context string from all result snippets
    all_content = []
    for resp in (skills_response, cert_response, market_response):
        for result in resp.get("results", []):
            snippet = result.get("content", "").strip()
            if snippet:
                all_content.append(snippet)

    combined_context = "\n\n".join(all_content[:8])  # cap to avoid huge payloads

    # ── Extract structured data from the answers ──────────────────────────────
    trends         = _extract_list(skills_answer, "trends",         job_title, combined_context)
    top_skills     = _extract_list(skills_answer, "skills",         job_title, combined_context)
    certifications = _extract_list(cert_answer,   "certifications", job_title, combined_context)
    salary_range   = _extract_salary(market_answer, combined_context)
    market_demand  = _extract_demand(market_answer, job_title, combined_context)

    return {
        "trends":         trends,
        "top_skills":     top_skills,
        "certifications": certifications,
        "salary_range":   salary_range,
        "market_demand":  market_demand,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_list(answer: str, kind: str, job_title: str, context: str) -> list:
    """
    Heuristically pull a list of items from a Tavily answer string.
    Falls back to sensible defaults derived from the context when the answer
    is too short or empty.
    """
    import re

    source = answer if len(answer) > 40 else context

    # Try to find bullet / numbered list items
    items = re.findall(r"(?:^|\n)\s*[-•*\d.]+\s+(.+)", source)
    items = [i.strip().rstrip(".,;") for i in items if len(i.strip()) > 3]

    if kind == "skills":
        # Also grab comma-separated skill lists
        comma_items = re.findall(r"(?:skills?|technologies|tools)[:\s]+([A-Za-z0-9 ,/#+.]+)", source, re.IGNORECASE)
        for chunk in comma_items:
            items += [s.strip() for s in chunk.split(",") if len(s.strip()) > 1]

    # Deduplicate while preserving order
    seen, unique = set(), []
    for item in items:
        key = item.lower()
        if key not in seen and len(item) < 80:
            seen.add(key)
            unique.append(item)

    if kind == "certifications":
        return unique[:8] if unique else _default_certs(job_title)
    if kind == "skills":
        return unique[:15] if unique else _default_skills(job_title)
    # trends
    return unique[:6] if unique else _default_trends(job_title)


def _extract_salary(answer: str, context: str) -> str:
    import re
    source = answer if len(answer) > 20 else context
    # Look for patterns like $120,000 – $160,000 or $120k-$160k
    match = re.search(
        r"\$[\d,]+[kK]?\s*[-–—to]+\s*\$[\d,]+[kK]?",
        source,
    )
    if match:
        return match.group(0).strip()
    # Fallback: look for any dollar figure
    match = re.search(r"\$[\d,]+[kK]?(?:\s*(?:per year|annually|/yr))?", source)
    if match:
        return match.group(0).strip()
    return "Salary data not available"


def _extract_demand(answer: str, job_title: str, context: str) -> str:
    source = answer if len(answer) > 40 else context
    # Return the first meaningful sentence that mentions demand/hiring/growth
    import re
    sentences = re.split(r"(?<=[.!?])\s+", source)
    keywords = ("demand", "hiring", "growth", "market", "outlook", "opportunit", "job")
    for sent in sentences:
        if any(kw in sent.lower() for kw in keywords) and len(sent) > 30:
            return sent.strip()
    # Fallback: first sentence of the answer
    if sentences:
        return sentences[0].strip()
    return f"Strong demand for {job_title} professionals across multiple industries."


# ── Static fallbacks (used when Tavily returns sparse data) ──────────────────

def _default_skills(job_title: str) -> list:
    title_lower = job_title.lower()
    if any(t in title_lower for t in ("ml", "machine learning", "ai", "data scientist")):
        return ["Python", "TensorFlow", "PyTorch", "Scikit-learn", "SQL", "MLOps",
                "Docker", "Kubernetes", "Statistics", "Deep Learning", "NLP",
                "Data Visualization", "Cloud (AWS/GCP/Azure)", "Git", "Jupyter"]
    if any(t in title_lower for t in ("embedded", "firmware", "rtos")):
        return ["C", "C++", "RTOS", "ARM Cortex", "CAN", "SPI", "I2C", "UART",
                "Python", "Git", "JTAG", "FreeRTOS", "Linux", "CMake", "Unit Testing"]
    if any(t in title_lower for t in ("frontend", "front-end", "react", "vue")):
        return ["JavaScript", "TypeScript", "React", "CSS", "HTML", "Next.js",
                "REST APIs", "Git", "Testing", "Webpack", "Accessibility", "CI/CD",
                "GraphQL", "Performance Optimization", "Responsive Design"]
    if any(t in title_lower for t in ("backend", "back-end", "node", "django")):
        return ["Python", "Node.js", "SQL", "PostgreSQL", "REST APIs", "Docker",
                "Kubernetes", "Redis", "Git", "CI/CD", "Microservices", "AWS",
                "Authentication", "System Design", "Testing"]
    # Generic software / default
    return ["Python", "SQL", "Git", "Docker", "REST APIs", "Agile", "CI/CD",
            "Cloud Platforms", "Communication", "Problem Solving", "Testing",
            "Linux", "Data Analysis", "System Design", "Collaboration"]


def _default_certs(job_title: str) -> list:
    title_lower = job_title.lower()
    if any(t in title_lower for t in ("cloud", "aws", "azure", "gcp")):
        return ["AWS Solutions Architect", "Google Cloud Professional", "Azure Administrator",
                "Kubernetes (CKA)", "Terraform Associate"]
    if any(t in title_lower for t in ("ml", "machine learning", "ai", "data")):
        return ["Google Professional ML Engineer", "AWS ML Specialty",
                "TensorFlow Developer Certificate", "Coursera Deep Learning Specialization",
                "Databricks Certified Associate"]
    if any(t in title_lower for t in ("security", "cyber")):
        return ["CISSP", "CEH", "CompTIA Security+", "OSCP", "AWS Security Specialty"]
    if "project" in title_lower or "manager" in title_lower:
        return ["PMP", "Scrum Master (CSM)", "PRINCE2", "PMI-ACP", "SAFe Agilist"]
    return ["Relevant Cloud Certification", "Industry-Specific Certification",
            "Agile / Scrum Certification", "Leadership & Management Certification"]


def _default_trends(job_title: str) -> list:
    return [
        f"Growing demand for {job_title} professionals with AI/ML exposure",
        "Remote and hybrid roles increasingly common",
        "Emphasis on cross-functional collaboration and communication skills",
        "Open-source contributions valued by top employers",
        "Continuous learning and upskilling expected",
        "Strong focus on measurable impact and quantified achievements",
    ]
