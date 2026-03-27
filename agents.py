"""
agents.py — Step 4
==================
Three sub-agents for the self-evolving security code review pipeline,
plus a VersionedPrompt class that tracks prompt history for the detector.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap: import the openai-agents SDK despite this file being named
# 'agents.py' (same as the installed SDK package).
#
# Python adds this module to sys.modules['agents'] BEFORE executing its body,
# which normally causes a circular import.  We work around this by:
#   1. Popping the partial self-reference from sys.modules
#   2. Removing the project root from sys.path so Python cannot find THIS file
#   3. Importing Agent/Runner/trace from the installed SDK
#   4. Restoring sys.path and sys.modules so nothing else breaks
# ---------------------------------------------------------------------------
_this_dir = str(Path(__file__).parent.resolve())
_self_module = sys.modules.pop("agents", None)

_evicted: list[tuple[int, str]] = []
for _i in range(len(sys.path) - 1, -1, -1):
    try:
        _resolved = str(Path(sys.path[_i] or ".").resolve())
    except Exception:
        continue
    if _resolved == _this_dir:
        _evicted.append((_i, sys.path.pop(_i)))

try:
    from agents import Agent, Runner, trace  # openai-agents SDK
finally:
    for _i, _p in reversed(_evicted):
        sys.path.insert(_i, _p)
    if _self_module is not None:
        sys.modules["agents"] = _self_module


# ---------------------------------------------------------------------------
# VersionedPrompt — tracks the full history of prompts for the detector agent
# ---------------------------------------------------------------------------

@dataclass
class PromptVersion:
    version: int
    prompt: str
    model: str
    timestamp: str
    metadata: dict = field(default_factory=dict)


class VersionedPrompt:
    """
    Maintains an immutable append-only history of prompt versions.
    The detector agent is reconstructed from .current() before each use.
    """

    def __init__(self, initial_prompt: str, model: str, metadata: Optional[dict] = None) -> None:
        self._history: list[PromptVersion] = []
        self._append(initial_prompt, model, metadata or {})

    def _append(self, prompt: str, model: str, metadata: dict) -> None:
        self._history.append(PromptVersion(
            version=len(self._history) + 1,
            prompt=prompt,
            model=model,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        ))

    def current(self) -> PromptVersion:
        """Return the latest prompt version."""
        return self._history[-1]

    def update(self, new_prompt: str, model: Optional[str] = None, metadata: Optional[dict] = None) -> None:
        """Append a new version with an updated prompt (and optionally a new model)."""
        self._append(
            new_prompt,
            model if model is not None else self.current().model,
            metadata or {},
        )

    def revert_to_version(self, n: int) -> None:
        """Create a new version whose prompt and model match version n."""
        target = next((v for v in self._history if v.version == n), None)
        if target is None:
            raise ValueError(
                f"Version {n} not found (available: 1–{len(self._history)})"
            )
        self._append(target.prompt, target.model, {"reverted_from_version": n})

    @property
    def history(self) -> list[PromptVersion]:
        return list(self._history)


# ---------------------------------------------------------------------------
# vulnerability_detector — evolves; wrapped in VersionedPrompt
# Intentionally weak starting prompt so the loop has room to improve it.
# ---------------------------------------------------------------------------

_DETECTOR_INITIAL_PROMPT = (
    "You are a security code reviewer. "
    "Given a code snippet, identify any security vulnerabilities."
)

vulnerability_detector_prompt = VersionedPrompt(
    initial_prompt=_DETECTOR_INITIAL_PROMPT,
    model="gpt-4.1-mini",
)


def make_detector_agent() -> Agent:
    """Construct a fresh Agent from the current prompt version."""
    v = vulnerability_detector_prompt.current()
    return Agent(
        name="vulnerability_detector",
        instructions=v.prompt,
        model=v.model,
    )


# ---------------------------------------------------------------------------
# severity_classifier — static, never evolves
# ---------------------------------------------------------------------------

severity_classifier = Agent(
    name="severity_classifier",
    instructions=(
        "You are a CVSS scoring assistant. Given a vulnerability description, "
        "assign a CVSS v3.1 base score. "
        "Output the score as a decimal number (e.g. 7.5) and one sentence of justification."
    ),
    model="gpt-4.1-mini",
)

# ---------------------------------------------------------------------------
# remediation_advisor — static, never evolves
# ---------------------------------------------------------------------------

remediation_advisor = Agent(
    name="remediation_advisor",
    instructions=(
        "You are a secure coding expert. Given a vulnerability and the original code, provide: "
        "1) The CWE identifier, "
        "2) A corrected version of the code, "
        "3) One sentence explaining why the fix works. "
        "Never suggest string sanitization for injection flaws. "
        "Always recommend parameterized queries for SQL. "
        "Always recommend allowlist validation for path traversal."
    ),
    model="gpt-4.1-mini",
)


# ---------------------------------------------------------------------------
# run_pipeline — orchestrates all three agents in sequence
# ---------------------------------------------------------------------------

async def run_pipeline(code: str, detector_agent: Agent) -> str:
    """
    Run the full security review pipeline:
      1. Detect vulnerabilities (evolving detector)
      2. Classify CVSS severity
      3. Advise remediation

    detector output → severity classifier → remediation advisor.
    Returns a single combined string.
    """
    with trace("security_review_pipeline"):
        detection_result = await Runner.run(
            detector_agent,
            f"Review this code for security vulnerabilities:\n\n{code}",
        )
        detection_output = detection_result.final_output

        severity_result = await Runner.run(
            severity_classifier,
            f"Vulnerability description:\n{detection_output}",
        )
        severity_output = severity_result.final_output

        remediation_result = await Runner.run(
            remediation_advisor,
            f"Vulnerability:\n{detection_output}\n\nOriginal code:\n{code}",
        )
        remediation_output = remediation_result.final_output

    return (
        f"[DETECTION]\n{detection_output}\n\n"
        f"---\n\n"
        f"[SEVERITY]\n{severity_output}\n\n"
        f"---\n\n"
        f"[REMEDIATION]\n{remediation_output}"
    )
