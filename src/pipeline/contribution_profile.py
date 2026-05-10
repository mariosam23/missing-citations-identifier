"""Stage 1.5 — Contribution profile extraction.

Extracts a structured summary of what a paper proposes (novel) versus what it
uses (prior work) from the title, abstract, introduction, and conclusion. The
profile is later injected into the classifier's system prompt so descriptive
sentences about the authors' novel contributions are not flagged as missing
citations.

This module is decoupled from the classifier: it produces a
``ContributionProfile`` and a paper-specific augmented system prompt. Wiring
into ``CitationClassifier`` happens at the call site via the existing
``system_prompt`` constructor parameter.
"""


import json
import re
from typing import Any, Protocol

from entities import ContributionProfile, ParsedPaper
from prompts import (
    CLASSIFIER_SYSTEM_PROMPT,
    CONTRIBUTION_PROFILE_SYSTEM_PROMPT,
    CONTRIBUTION_PROFILE_USER_PROMPT_TEMPLATE,
    NOVEL_WORK_GUARD_EXAMPLE,
    NOVEL_WORK_GUARD_RULE,
)
from utils import logger
from utils.config import config


_INTRO_RE = re.compile(r"^\s*(\d+\.?\s*)?introduction\b", re.IGNORECASE)
_CONCLUSION_RE = re.compile(
    r"^\s*(\d+\.?\s*)?(conclusions?|discussion and conclusions?|summary)\b",
    re.IGNORECASE,
)


class CompletionClient(Protocol):
    def complete(
        self,
        system: str,
        user: str,
        response_mime_type: str | None = None,
    ) -> str:
        ...


class ContributionProfileExtractor:
    """One-shot LLM call that extracts a paper's contribution profile."""

    def __init__(
        self,
        model: str | None = None,
        client: CompletionClient | None = None,
        system_prompt: str | None = None,
        max_chars_per_section: int = 6000,
    ) -> None:
        model = model or config.CLASSIFIER_MODEL
        if client is None:
            from llm.gemini_client import GeminiClient

            client = GeminiClient(model=model, temperature=0.1, max_tokens=1500)
        self.client = client
        self.system_prompt = system_prompt or CONTRIBUTION_PROFILE_SYSTEM_PROMPT
        self.max_chars_per_section = max_chars_per_section

    def extract(self, paper: ParsedPaper) -> ContributionProfile:
        chosen_sections = self._select_sections(paper)
        sections_block = self._format_sections(chosen_sections)
     
        user_prompt = CONTRIBUTION_PROFILE_USER_PROMPT_TEMPLATE.format(
            title=paper.title,
            abstract=paper.abstract,
            sections_block=sections_block,
        )

        logger.info(
            "Extracting contribution profile from %d section(s): %s",
            len(chosen_sections),
            list(chosen_sections.keys()),
        )
        response = self.client.complete(
            self.system_prompt,
            user_prompt,
            response_mime_type="application/json",
        )
        logger.debug("Contribution profile raw response:\n%s", response)

        parsed = self._parse_json_object(response)
        return ContributionProfile(
            system_names=_str_tuple(parsed.get("system_names")),
            novel_contributions=_str_tuple(parsed.get("novel_contributions")),
            uses=_str_tuple(parsed.get("uses")),
            proposes=_str_tuple(parsed.get("proposes")),
            source_sections=tuple(chosen_sections.keys()),
        )

    def _format_sections(self, sections: dict[str, str]) -> str:
        if not sections:
            return ""
        parts: list[str] = []
        for name, text in sections.items():
            truncated = text[: self.max_chars_per_section]
            parts.append(f"--- {name} ---\n{truncated}")
        return "\n\n".join(parts)

    @staticmethod
    def _select_sections(paper: ParsedPaper) -> dict[str, str]:
        intro_key: str | None = None
        conclusion_key: str | None = None
        for name in paper.sections:
            if intro_key is None and _INTRO_RE.match(name):
                intro_key = name
            if _CONCLUSION_RE.match(name):
                conclusion_key = name

        chosen: dict[str, str] = {}
        if intro_key:
            chosen[intro_key] = paper.sections[intro_key]
        if conclusion_key and conclusion_key != intro_key:
            chosen[conclusion_key] = paper.sections[conclusion_key]
        if chosen:
            return chosen

        if paper.sections:
            keys = list(paper.sections.keys())
            chosen[keys[0]] = paper.sections[keys[0]]
            if len(keys) > 1:
                chosen[keys[-1]] = paper.sections[keys[-1]]
        return chosen

    @staticmethod
    def _parse_json_object(response: str) -> dict[str, Any]:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            json_match = re.search(r"\{[\s\S]*\}", cleaned)
            if not json_match:
                raise ValueError(
                    f"Could not parse contribution profile response as JSON: {response}"
                )
            parsed = json.loads(json_match.group())

        if not isinstance(parsed, dict):
            raise ValueError(
                f"Expected contribution profile response to be a JSON object, got: {type(parsed).__name__}"
            )
        return parsed


def _str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return tuple(out)


# ---------------------------------------------------------------------------
# Classifier-prompt augmentation
# ---------------------------------------------------------------------------


def format_profile_block(profile: ContributionProfile) -> str:
    """Render a ContributionProfile as a human-readable block for prompt injection."""
    lines = ["Paper Contribution Profile:"]
    if profile.proposes:
        lines.append(
            '- Proposed in this paper (descriptive sentences about these default to NOT_CITATION_WORTHY): '
            + "; ".join(profile.proposes)
        )
    if profile.novel_contributions:
        lines.append(
            "- Novel contributions: " + "; ".join(profile.novel_contributions)
        )
    if profile.uses:
        lines.append(
            "- External building blocks (still need cites if mentioned): "
            + "; ".join(profile.uses)
        )
    return "\n".join(lines)


def build_classifier_system_prompt(
    profile: ContributionProfile,
    base_prompt: str | None = None,
) -> str:
    """Return a paper-specific classifier system prompt augmented with the profile.

    The base ``CLASSIFIER_SYSTEM_PROMPT`` is unchanged. We prepend the profile
    block, append the novel-work guard rule, and append a worked example so the
    classifier knows how to apply the rule. Pass the result to
    ``CitationClassifier(system_prompt=...)``.
    """
    base = base_prompt or CLASSIFIER_SYSTEM_PROMPT
    profile_block = format_profile_block(profile)

    return (
        f"{profile_block}\n\n"
        f"{base.rstrip()}\n\n"
        f"ADDITIONAL RULE:\n{NOVEL_WORK_GUARD_RULE}\n\n"
        f"{NOVEL_WORK_GUARD_EXAMPLE}\n"
    )
