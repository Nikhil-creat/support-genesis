"""Project attribution metadata — surfaced on the API root endpoint and in
generated reports/docs. Single source of truth so the credit line never
drifts out of sync across files.
"""
from __future__ import annotations

PROJECT_NAME = "Autonomous Customer Support & Action Engine"
PROJECT_TAGLINE = (
    "Enterprise-grade agentic AI platform — multi-agent orchestration, hybrid RAG, "
    "CNN-based visual inspection, MCP tool integration, and HITL-governed automation."
)

AUTHOR_NAME = "Nikhil Chary Sriramoju"
AUTHOR_CREDIT_LINE = f"Designed and Developed by {AUTHOR_NAME.upper()}"

AUTHOR_LINKS = {
    "github": "https://github.com/Nikhil-creat",
    "linkedin": "https://in.linkedin.com/in/nikhil-chary-sriramoju-95041b38a",
    "email": "sriramojunikhil66@gmail.com",
    "instagram": "https://www.instagram.com/nikhil__sriramoju",
}


def branding_payload() -> dict:
    return {
        "project": PROJECT_NAME,
        "tagline": PROJECT_TAGLINE,
        "credit": AUTHOR_CREDIT_LINE,
        "author": AUTHOR_NAME,
        "links": AUTHOR_LINKS,
    }
