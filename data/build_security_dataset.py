"""
Security Dataset Pipeline
=========================
Transforms CVEfixes (CIRCL/vulnerability-cwe-patch on HuggingFace) into a
clean CSV ready for the self-evolving security review agent.

Output schema per row:
  - code            : str   vulnerable function / diff snippet
  - safe_fix        : str   patched version of the same code
  - language        : str   detected language (python, javascript, c, etc.)
  - known_cwes      : list  e.g. ["CWE-89", "CWE-79"]
  - cve_id          : str   e.g. "CVE-2023-1234"
  - expected_severity: list  [low_score, high_score] from NVD CVSS v3.1
  - is_vulnerable   : bool  True for vuln rows, False for safe rows
  - source          : str   which dataset it came from

Usage:
  pip install datasets requests pandas tqdm
  python build_security_dataset.py

Optional flags (edit CONFIG below):
  - MAX_VULN_ROWS      cap on vulnerable rows to fetch
  - MAX_SAFE_ROWS      cap on safe (negative) rows
  - LANGUAGES          filter to specific languages
  - OUTPUT_PATH        where to write the CSV
  - NVD_API_KEY        optional — raises NVD rate limit from 5/30s to 50/30s
"""

import base64
import json
import re
import os
import time
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm

_DATA_DIR = Path(__file__).parent

load_dotenv()

nvd_key = os.getenv("NVD_API_KEY")
# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------
CONFIG = {

    "MAX_VULN_ROWS": 500,       # vulnerable examples to pull
    "MAX_SAFE_ROWS": 500,       # safe (patched) examples to pull
    "LANGUAGES": None,          # None = all; or e.g. ["python", "javascript", "c"]
    "OUTPUT_PATH": str(_DATA_DIR / "security_agent_dataset.csv"),
    "NVD_API_KEY": nvd_key,        # set to your key string for higher rate limits
    "NVD_DELAY_S": 6,           # seconds between NVD calls (5/30s without key)
    "CVSS_TOLERANCE": 1.5,      # ± band around base score for expected_severity
    "MIN_CODE_CHARS": 80,       # skip trivially short snippets
    "MAX_CODE_CHARS": 8000,     # skip gigantic functions (context window)
    "RANDOM_SEED": 42,
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language detection from file extension in patch URL or commit message
# ---------------------------------------------------------------------------
EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".c": "c", ".cpp": "cpp", ".cc": "cpp", ".h": "c",
    ".java": "java", ".go": "go", ".rb": "ruby", ".php": "php",
    ".rs": "rust", ".cs": "csharp", ".swift": "swift", ".kt": "kotlin",
}

def detect_language(patch_url: str, commit_msg: str) -> str:
    url_lower = (patch_url or "").lower()
    for ext, lang in EXT_TO_LANG.items():
        if url_lower.endswith(ext) or f"{ext}." in url_lower:
            return lang
    return "unknown"


# ---------------------------------------------------------------------------
# Patch decoder — CVEfixes stores diffs as base64 unified diffs
# ---------------------------------------------------------------------------
def decode_patch(patch_b64: str) -> str:
    """Decode base64 patch text."""
    try:
        return base64.b64decode(patch_b64).decode("utf-8", errors="replace")
    except Exception:
        return ""


def split_diff(diff_text: str) -> tuple[str, str]:
    """
    Extract 'before' (vulnerable) and 'after' (fixed) code from a unified diff.
    Returns (before_code, after_code) as plain strings.
    Lines prefixed '-' are removed in the fix; lines prefixed '+' are added.
    """
    before_lines, after_lines = [], []
    for line in diff_text.splitlines():
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        if line.startswith("-"):
            before_lines.append(line[1:])
        elif line.startswith("+"):
            after_lines.append(line[1:])
        else:
            # Context line — appears in both
            before_lines.append(line)
            after_lines.append(line)
    return "\n".join(before_lines), "\n".join(after_lines)


# ---------------------------------------------------------------------------
# NVD CVSS lookup with rate limiting and caching
# ---------------------------------------------------------------------------
_nvd_cache: dict[str, Optional[float]] = {}

def get_cvss_score(cve_id: str, api_key: Optional[str], delay: float) -> Optional[float]:
    """
    Fetch CVSS v3.1 base score from NVD for a given CVE ID.
    Returns None if unavailable. Caches results in-process.
    """
    if cve_id in _nvd_cache:
        return _nvd_cache[cve_id]

    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
    try:
        time.sleep(delay)
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            _nvd_cache[cve_id] = None
            return None

        metrics = vulns[0]["cve"].get("metrics", {})
        # Prefer v3.1, fall back to v3.0, then v2
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if key in metrics and metrics[key]:
                score = metrics[key][0]["cvssData"]["baseScore"]
                _nvd_cache[cve_id] = float(score)
                return float(score)
    except Exception as e:
        log.debug(f"NVD lookup failed for {cve_id}: {e}")

    _nvd_cache[cve_id] = None
    return None


def score_to_range(score: float, tol: float) -> list[float]:
    """Convert a point score to [low, high] range with tolerance."""
    return [round(max(0.0, score - tol), 1), round(min(10.0, score + tol), 1)]


# ---------------------------------------------------------------------------
# CWE normaliser — ensure consistent "CWE-NNN" format
# ---------------------------------------------------------------------------
def normalise_cwes(raw: object) -> list[str]:
    """
    Accept various CWE representations and return a clean list.
    Input may be a string, list of strings, or list of dicts.
    """
    if not raw:
        return []
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        return []

    result = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("cweId") or item.get("id") or item.get("name") or ""
        else:
            text = str(item)
        # Extract numeric part and reformat
        m = re.search(r"(\d+)", text)
        if m:
            result.append(f"CWE-{m.group(1)}")
    return list(set(result)) or []


# ---------------------------------------------------------------------------
# Row builder from a CIRCL patch entry
# ---------------------------------------------------------------------------
def build_rows_from_entry(entry: dict, cfg: dict) -> list[dict]:
    """
    One CIRCL entry can have multiple patches.
    Returns a list of row dicts (one per usable patch).
    """
    rows = []
    cve_id = entry.get("id", "")
    description = entry.get("description", "")
    cwes = normalise_cwes(entry.get("cwe", []))

    if not cwes:
        return []  # skip entries with no CWE label

    for patch in entry.get("patches", []):
        patch_b64 = patch.get("patch_text_b64", "")
        patch_url = patch.get("url", "")
        commit_msg = patch.get("commit_message", "")

        if not patch_b64:
            continue

        diff_text = decode_patch(patch_b64)
        if not diff_text:
            continue

        before_code, after_code = split_diff(diff_text)

        # Length filters
        if len(before_code) < cfg["MIN_CODE_CHARS"] or len(before_code) > cfg["MAX_CODE_CHARS"]:
            continue
        if len(after_code) < cfg["MIN_CODE_CHARS"]:
            continue

        lang = detect_language(patch_url, commit_msg)
        if cfg["LANGUAGES"] and lang not in cfg["LANGUAGES"]:
            continue

        rows.append({
            "code": before_code,
            "safe_fix": after_code,
            "language": lang,
            "known_cwes": cwes,
            "cve_id": cve_id,
            "expected_severity": None,   # filled later via NVD
            "is_vulnerable": True,
            "source": "CIRCL/vulnerability-cwe-patch",
            "description": description,
        })

    return rows


# ---------------------------------------------------------------------------
# Safe (negative) row generator
# "After" code from a real patch is the best negative example:
#   - Structurally similar to the vulnerable version
#   - Verified safe by the patch author
#   - Keeps language distribution realistic
# ---------------------------------------------------------------------------
def make_safe_rows(vuln_rows: list[dict], n: int, seed: int) -> list[dict]:
    """
    For each sampled vuln row, emit its patched version as a safe example.
    Resets is_vulnerable=False and clears known_cwes / expected_severity.
    """
    import random
    rng = random.Random(seed)
    sample = rng.sample(vuln_rows, min(n, len(vuln_rows)))

    safe_rows = []
    for row in sample:
        safe_rows.append({
            "code": row["safe_fix"],          # patched code is the input
            "safe_fix": row["safe_fix"],       # same — there's no change needed
            "language": row["language"],
            "known_cwes": [],                  # safe code has no CWEs
            "cve_id": row["cve_id"] + "_safe",
            "expected_severity": [0.0, 0.0],  # no severity for safe code
            "is_vulnerable": False,
            "source": row["source"] + " (safe variant)",
            "description": "Safe (patched) version of " + row["cve_id"],
        })
    return safe_rows


# ---------------------------------------------------------------------------
# CVSS enrichment pass — batch NVD lookups with progress bar
# ---------------------------------------------------------------------------
def enrich_with_cvss(rows: list[dict], cfg: dict) -> list[dict]:
    """
    For each vulnerable row, look up CVSS from NVD and set expected_severity.
    Rows without a CVSS score get a wide default range [0.0, 10.0].
    """
    unique_cves = list({r["cve_id"] for r in rows if r["is_vulnerable"]})
    log.info(f"Fetching CVSS for {len(unique_cves)} unique CVEs from NVD…")

    for cve_id in tqdm(unique_cves, desc="NVD CVSS lookup"):
        get_cvss_score(cve_id, cfg["NVD_API_KEY"], cfg["NVD_DELAY_S"])

    for row in rows:
        if not row["is_vulnerable"]:
            continue
        score = _nvd_cache.get(row["cve_id"])
        if score is not None:
            row["expected_severity"] = score_to_range(score, cfg["CVSS_TOLERANCE"])
        else:
            row["expected_severity"] = [0.0, 10.0]   # wide fallback

    return rows


# ---------------------------------------------------------------------------
# Quality filter — remove rows unsuitable for the agent graders
# ---------------------------------------------------------------------------
def quality_filter(rows: list[dict]) -> list[dict]:
    """
    Drop rows that would give misleading grader signals:
    - No CWE on a vulnerable row
    - Before and after code are identical (patch had no code changes)
    - Code is clearly binary/minified (no whitespace, very long lines)
    """
    kept = []
    for row in rows:
        if row["is_vulnerable"] and not row["known_cwes"]:
            continue
        if row["code"].strip() == row["safe_fix"].strip():
            continue
        # Heuristic: minified code has very long lines
        max_line = max((len(l) for l in row["code"].splitlines()), default=0)
        if max_line > 500:
            continue
        kept.append(row)
    return kept


# ---------------------------------------------------------------------------
# Serialise lists to JSON strings for CSV compatibility
# ---------------------------------------------------------------------------
def serialise_row(row: dict) -> dict:
    out = row.copy()
    out["known_cwes"] = json.dumps(row["known_cwes"])
    out["expected_severity"] = json.dumps(row["expected_severity"])
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    cfg = CONFIG
    log.info("Loading CIRCL/vulnerability-cwe-patch from HuggingFace…")
    ds = load_dataset("CIRCL/vulnerability-cwe-patch", split="train")
    log.info(f"Dataset loaded: {len(ds)} entries")

    # --- Build vulnerable rows ---
    log.info("Extracting vulnerable rows…")
    vuln_rows: list[dict] = []
    for entry in tqdm(ds, desc="Processing entries"):
        if len(vuln_rows) >= cfg["MAX_VULN_ROWS"]:
            break
        vuln_rows.extend(build_rows_from_entry(entry, cfg))

    log.info(f"Extracted {len(vuln_rows)} vulnerable rows before filtering")
    vuln_rows = quality_filter(vuln_rows)
    vuln_rows = vuln_rows[: cfg["MAX_VULN_ROWS"]]
    log.info(f"After quality filter: {len(vuln_rows)} vulnerable rows")

    # --- Build safe rows from patched code ---
    log.info("Generating safe (negative) rows from patched code…")
    safe_rows = make_safe_rows(vuln_rows, cfg["MAX_SAFE_ROWS"], cfg["RANDOM_SEED"])
    log.info(f"Generated {len(safe_rows)} safe rows")

    # --- Merge ---
    all_rows = vuln_rows + safe_rows

    # --- CVSS enrichment (NVD API) ---
    log.info("Enriching vulnerable rows with CVSS scores from NVD…")
    all_rows = enrich_with_cvss(all_rows, cfg)

    # --- Serialise and write ---
    log.info(f"Writing {len(all_rows)} rows to {cfg['OUTPUT_PATH']}…")
    serialised = [serialise_row(r) for r in all_rows]

    # Drop internal-only description column
    for r in serialised:
        r.pop("description", None)

    df = pd.DataFrame(serialised)

    # Shuffle so vuln/safe rows are interleaved
    df = df.sample(frac=1, random_state=cfg["RANDOM_SEED"]).reset_index(drop=True)

    df.to_csv(cfg["OUTPUT_PATH"], index=False)
    log.info(f"Done. Dataset saved to: {cfg['OUTPUT_PATH']}")

    # --- Summary ---
    print("\n--- Dataset summary ---")
    print(f"Total rows      : {len(df)}")
    print(f"Vulnerable rows : {df['is_vulnerable'].sum()}")
    print(f"Safe rows       : {(~df['is_vulnerable']).sum()}")
    print(f"Languages       :")
    print(df["language"].value_counts().to_string())
    print(f"\nSample row (vulnerable):")
    vuln_sample = df[df["is_vulnerable"] == True].iloc[0]
    print(f"  cve_id          : {vuln_sample['cve_id']}")
    print(f"  known_cwes      : {vuln_sample['known_cwes']}")
    print(f"  expected_severity: {vuln_sample['expected_severity']}")
    print(f"  language        : {vuln_sample['language']}")
    print(f"  code (first 120): {vuln_sample['code'][:120].strip()}…")


if __name__ == "__main__":
    main()
