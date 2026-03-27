# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run dataset validation:**
```bash
.venv/Scripts/python data/dataset_loader.py data/security_agent_dataset_clean.csv
```

**Build raw dataset from scratch (requires NVD API key, takes ~30 min due to rate limiting):**
```bash
.venv/Scripts/python data/build_security_dataset.py
```

**Clean raw dataset:**
```bash
.venv/Scripts/python data/clean_data.py
```

**Create OpenAI Eval (run once, saves EVAL_ID to .env):**
```bash
.venv/Scripts/python data/setup_eval.py
```

## Environment Setup

Requires `.env` with:
- `OPENAI_API_KEY` — for OpenAI API calls and LLM-based grading
- `NVD_API_KEY` — optional; increases NVD rate limit from 5/30s to 50/30s for CVSS enrichment
- `EVAL_ID` — auto-saved after running `setup_eval.py`

Python 3.12, dependencies installed in `.venv/`. Key packages: `openai`, `pandas`, `datasets` (HuggingFace), `requests`, `tqdm`, `python-dotenv`.

## Architecture

This is a **data pipeline and evaluation framework** for a self-evolving security code review agent. The pipeline runs in four sequential steps:

```
CIRCL/vulnerability-cwe-patch (Hugging Face)
        ↓
build_security_dataset.py   → security_agent_dataset.csv       (~184k rows)
        ↓
clean_data.py               → security_agent_dataset_clean.csv (~156k rows)
        ↓
dataset_loader.py           → train/val/test splits (70/15/15, stratified)
        ↓
setup_eval.py               → OpenAI Eval with 4 graders
```

### Dataset Schema

Each row has 8 fields:
```python
{
    "code": str,               # vulnerable or safe function snippet
    "safe_fix": str,           # verified real-world patch from CVEfixes
    "language": str,           # detected from file extension
    "known_cwes": list,        # e.g. ["CWE-89", "CWE-79"]
    "cve_id": str,             # e.g. "CVE-2023-1234"
    "expected_severity": list, # [low, high] CVSS v3.1 score range from NVD
    "is_vulnerable": bool,     # True = vulnerable, False = safe variant
    "source": str              # dataset origin tag
}
```

For every vulnerable example, a corresponding safe variant is generated using the real patch — giving the agent both positive and negative examples.

### Evaluation (setup_eval.py)

Four graders are registered as an OpenAI Eval:

1. **CWE Coverage** (deterministic Python) — score = matched_cwes / total_known_cwes, pass ≥ 0.85
2. **False Positive** (deterministic Python) — penalizes security alarms on safe code, pass ≥ 0.80
3. **CVSS Severity Accuracy** (deterministic Python) — checks predicted score is within NVD range, pass ≥ 0.80
4. **Remediation Quality** (LLM-as-judge via `gpt-4.1`) — evaluates whether proposed fix addresses root cause using real CVE patches as ground truth, pass ≥ 0.80

### Key Implementation Notes

- `build_security_dataset.py` CONFIG dict controls dataset size (`MAX_VULN_ROWS`, `MAX_SAFE_ROWS`), language filter, and code length bounds.
- NVD API calls are rate-limited (`NVD_DELAY_S = 6`); a key reduces delay significantly.
- `dataset_loader.py` deserializes JSON-encoded list fields (`known_cwes`, `expected_severity`) when loading the CSV.
- Quality filtering removes: rows missing CWEs, identical before/after diffs, minified code (any line > 500 chars), and code outside `MIN_CODE_CHARS`/`MAX_CODE_CHARS` bounds.
