"""
setup_eval.py  —  Step 3
========================
Creates the OpenAI Eval with all four graders for the
self-evolving security code review agent.

Run once. Saves the eval_id to .env so every other script
can load it automatically.

Usage:
    python setup_eval.py

Requires in .env:
    OPENAI_API_KEY=sk-...
"""

import os
from dotenv import load_dotenv, set_key
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Data source schema
# Every eval run will pass rows with these fields:
#   item   = ground truth  (from your CSV)
#   sample = agent output  (what the agent produced)
# ---------------------------------------------------------------------------
data_source_config = {
    "type": "custom",
    "item_schema": {
        "type": "object",
        "properties": {
            "code":              {"type": "string"},   # vulnerable code snippet
            "safe_fix":         {"type": "string"},   # verified patched version
            "known_cwes":       {"type": "string"},   # JSON array e.g. '["CWE-89"]'
            "is_vulnerable":    {"type": "boolean"},  # True / False
            "expected_severity":{"type": "string"},   # JSON array e.g. '[6.0, 9.0]'
        },
        "required": ["code", "safe_fix", "known_cwes", "is_vulnerable", "expected_severity"],
    },
    "include_sample_schema": False,
}

# ---------------------------------------------------------------------------
# Grader 1 — CWE Coverage (Python, deterministic)
# Did the agent name every CWE that exists in this snippet?
# Score = matched_cwes / total_known_cwes
# ---------------------------------------------------------------------------
cwe_coverage_grader = {
    "type": "python",
    "name": "cwe_coverage_grader",
    "image_tag": "2025-05-08",
    "pass_threshold": 0.85,
    "source": r"""
import re, json

def grade(sample: dict, item: dict) -> float:
    output = sample.get("output_text", "").lower()

    # known_cwes is stored as a JSON string in the CSV
    raw = item.get("known_cwes", "[]")
    try:
        known_cwes = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        known_cwes = []

    # Safe code rows have no CWEs — nothing to check
    if not known_cwes:
        return 1.0

    # Extract all CWE-NNN patterns from agent output
    detected = set(re.findall(r"cwe-\d+", output, re.IGNORECASE))
    known_set = set(c.lower() for c in known_cwes)

    matched = len(known_set & detected)
    return round(matched / len(known_set), 4)
""",
}

# ---------------------------------------------------------------------------
# Grader 2 — False Positive Penalty (Python, deterministic)
# Does the agent raise a security alarm on code that is actually safe?
# Score = 1.0 if agent is correct, 0.0 if it false-alarms on safe code
# ---------------------------------------------------------------------------
false_positive_grader = {
    "type": "python",
    "name": "false_positive_grader",
    "image_tag": "2025-05-08",
    "pass_threshold": 0.80,
    "source": r"""
def grade(sample: dict, item: dict) -> float:
    is_vulnerable = item.get("is_vulnerable", True)
    output = sample.get("output_text", "").lower()

    # Phrases that indicate the agent is raising an alarm
    alarm_phrases = [
        "vulnerability", "vulnerable", "cwe-",
        "injection", "exploit", "attack",
        "insecure", "unsafe", "risk", "malicious",
        "path traversal", "cross-site", "overflow",
    ]
    raised_alarm = any(p in output for p in alarm_phrases)

    if is_vulnerable:
        # Vulnerable code: we WANT the agent to raise an alarm
        # (CWE grader handles whether it named the right CWE)
        return 1.0
    else:
        # Safe code: penalise false alarms
        if raised_alarm:
            return 0.0
        else:
            return 1.0
""",
}

# ---------------------------------------------------------------------------
# Grader 3 — CVSS Severity Accuracy (Python, deterministic)
# Is the agent's stated severity score within the expected range?
# ---------------------------------------------------------------------------
cvss_accuracy_grader = {
    "type": "python",
    "name": "cvss_accuracy_grader",
    "image_tag": "2025-05-08",
    "pass_threshold": 0.80,
    "source": r"""
import re, json

def grade(sample: dict, item: dict) -> float:
    # Safe code has no severity to check
    if not item.get("is_vulnerable", True):
        return 1.0

    output = sample.get("output_text", "")

    # expected_severity is stored as a JSON string e.g. "[6.0, 9.0]"
    raw = item.get("expected_severity", "[0.0, 10.0]")
    try:
        expected = json.loads(raw) if isinstance(raw, str) else raw
        low, high = float(expected[0]), float(expected[1])
    except Exception:
        return 1.0  # can't validate, don't penalise

    # Wide fallback range means NVD had no data — skip grading
    if low == 0.0 and high == 10.0:
        return 1.0

    # Find the first decimal number in the output (e.g. "7.5", "CVSS: 8.1")
    match = re.search(r"\b(\d+\.\d)\b", output)
    if not match:
        # Agent gave no score at all
        return 0.0

    score = float(match.group(1))
    if low <= score <= high:
        return 1.0

    # Partial credit — penalise proportionally to how far off it is
    deviation = min(abs(score - low), abs(score - high))
    return round(max(0.0, 1.0 - (deviation / 3.0)), 4)
""",
}

# ---------------------------------------------------------------------------
# Grader 4 — Remediation Quality (LLM-as-judge)
# Does the fix actually address the root cause?
# Anchored to the real patch from the CVE — prevents confident-but-wrong fixes
# ---------------------------------------------------------------------------
remediation_judge_grader = {
    "type": "score_model",
    "name": "remediation_quality_judge",
    "model": "gpt-4.1",
    "pass_threshold": 0.80,
    "input": [
        {
            "role": "system",
            "content": (
                "You are a senior application security engineer evaluating "
                "AI-generated vulnerability remediation advice.\n\n"
                "Score from 0.0 to 1.0 using this rubric:\n\n"
                "1.0  — Fix addresses the root cause. Correct API or pattern "
                "recommended. No new vulnerabilities introduced. "
                "Language-idiomatic.\n"
                "0.75 — Root cause addressed but fix is verbose, has minor "
                "gaps, or uses a non-idiomatic approach that still works.\n"
                "0.5  — Partial: addresses symptoms but not root cause, or "
                "correct concept but wrong implementation detail.\n"
                "0.25 — Fix is in the right direction but would not prevent "
                "exploitation.\n"
                "0.0  — Fix is wrong, introduces new vulnerabilities, or is "
                "missing entirely.\n\n"
                "Be especially strict about:\n"
                "- SQL injection: parameterized queries only, NOT string sanitization\n"
                "- XSS: context-aware output encoding, NOT just html escaping\n"
                "- Path traversal: allowlist validation only, NOT blacklist filtering\n"
                "- Crypto: established libraries only, NEVER roll-your-own\n\n"
                "Respond with a single number between 0 and 1."
            ),
        },
        {
            "role": "user",
            "content": (
                "Vulnerable code:\n{{item.code}}\n\n"
                "Verified real-world fix (from the CVE patch):\n{{item.safe_fix}}\n\n"
                "Agent output:\n{{sample.output_text}}"
            ),
        },
    ],
    "range": [0, 1],
}

# ---------------------------------------------------------------------------
# Create the eval
# ---------------------------------------------------------------------------
print("Creating eval with 4 graders...")

eval_obj = client.evals.create(
    name="security_agent_eval_v1",
    data_source_config=data_source_config,
    testing_criteria=[
        cwe_coverage_grader,
        false_positive_grader,
        cvss_accuracy_grader,
        remediation_judge_grader,
    ],
)

eval_id = eval_obj.id
print(f"\nEval created successfully!")
print(f"Eval ID: {eval_id}")

# ---------------------------------------------------------------------------
# Save eval_id to .env so other scripts can load it automatically
# ---------------------------------------------------------------------------
env_path = os.path.join(os.path.dirname(__file__), ".env")
if not os.path.exists(env_path):
    # if .env is one level up (project root), find it
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")

set_key(env_path, "EVAL_ID", eval_id)
print(f"\nEVAL_ID saved to .env: {env_path}")
print("\nNext: run setup_agents.py  (Step 4)")