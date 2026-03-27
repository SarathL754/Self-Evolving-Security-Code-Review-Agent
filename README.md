# Self-Evolving Security Code Review Agent

A self-improving AI agent that automatically gets better at detecting software vulnerabilities. It starts with a deliberately weak prompt, runs against real CVE examples, grades itself using four automated metrics, and rewrites its own detection prompt whenever it underperforms — then validates the result on a held-out split to confirm genuine improvement.

---

## How It Works

The system is a closed loop:

```
[Code snippet (CVE)]
        ↓
 vulnerability_detector   ← evolving prompt (starts weak, improves automatically)
        ↓
 severity_classifier      ← assigns CVSS v3.1 base score (static)
        ↓
 remediation_advisor      ← provides CWE, fixed code, explanation (static)
        ↓
 OpenAI Eval (4 graders)  ← scores the full pipeline output
        ↓
 metaprompt_agent         ← rewrites the detector prompt on failure
        ↓
 VersionedPrompt          ← appends new version, tracks history
        ↓
 validation split         ← selects the best version on unseen data
        ↓
 best_prompt.json         ← persisted for final evaluation
```

After training, `evaluate.py` runs the initial weak prompt (v1) and the best evolved prompt (vN) side-by-side on the held-out test split and prints a before/after comparison table.

---

## Project Structure

```
self-evolving/
├── agents.py                   # Three sub-agents + VersionedPrompt class
├── evolving_loop.py            # Self-evolving training loop (Step 5)
├── evaluate.py                 # Before-vs-after evaluation on test split
├── best_prompt.json            # Best evolved prompt saved by evolving_loop.py
├── task.md                     # Original project specification
│
├── data/
│   ├── build_security_dataset.py   # Step 1: pull CVEfixes from HuggingFace + NVD
│   ├── clean_data.py               # Step 2: strip mail headers, filter short rows
│   ├── dataset_loader.py           # Step 3: load CSV, deserialise, split train/val/test
│   └── setup_eval.py               # Step 3: register OpenAI Eval with 4 graders
│
└── results/
    └── eval_results_<timestamp>.json   # Raw before/after metrics (written by evaluate.py)
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd self-evolving
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

`.env` contents:

```
OPENAI_API_KEY=sk-...          # Required
NVD_API_KEY=                   # Optional — raises NVD rate limit (5→50 req/30s)
EVAL_ID=                       # Auto-written by data/setup_eval.py
```

### 3. Build the dataset (one-time, ~30 min without NVD key)

```bash
python data/build_security_dataset.py   # → data/security_agent_dataset.csv
python data/clean_data.py               # → data/security_agent_dataset_clean.csv
```

### 4. Register the OpenAI Eval (one-time)

```bash
python data/setup_eval.py               # writes EVAL_ID to .env
```

### 5. Run the self-evolving loop

```bash
python evolving_loop.py                 # trains on 60 rows, saves best_prompt.json
```

### 6. Evaluate before vs. after

```bash
python evaluate.py                      # default: 50 test rows
python evaluate.py --rows 100           # use more rows for a stronger signal
```

Sample output:

```
============================================================================
  SECURITY AGENT — BEFORE vs AFTER EVALUATION
  50 held-out test rows  |  never seen during training
============================================================================
  Metric                            BEFORE (v1)    AFTER (v7)  DELTA
----------------------------------------------------------------------------
  Overall avg score                      0.4821        0.7134  +0.2313  (+47.9%)
  Pass rate (lenient)                    32.0 %        74.0 %  +42.0 pp
----------------------------------------------------------------------------
  Grader                            avg  pass%    avg  pass%  avg delta
----------------------------------------------------------------------------
  CWE Coverage                    0.312    18%   0.681    61%  +0.3690
  False Positive Penalty          0.720    72%   0.860    86%  +0.1400
  CVSS Accuracy                   0.551    55%   0.742    74%  +0.1910
  Remediation Quality             0.446    35%   0.618    62%  +0.1720
============================================================================
```

---

## Components

### `agents.py` — Three sub-agents

| Agent | Role | Evolves? |
|---|---|---|
| `vulnerability_detector` | Identify vulnerabilities and CWEs in code | Yes — prompt rewritten by metaprompt agent |
| `severity_classifier` | Assign CVSS v3.1 base score | No |
| `remediation_advisor` | Provide CWE ID, fixed code, fix explanation | No |

The detector is wrapped in a `VersionedPrompt` object that maintains an immutable append-only history. Each version stores: `version`, `prompt`, `model`, `timestamp`, and optional `metadata`. The loop can call `.update()` to append a new version or `.revert_to_version(n)` to roll back.

### `evolving_loop.py` — Self-evolving loop

Key design decisions:

- **Failure window**: the prompt only evolves after **3 consecutive failures** — this prevents overfitting to a single hard example.
- **Placeholder guard**: evolved prompts containing stub patterns like `[Insert code here]` are rejected and the previous version is kept.
- **Validation-based selection**: at the end of training, every prompt version is scored on a held-out validation split, and the version with the best validation average is saved to `best_prompt.json` — not necessarily the final version.
- **Metaprompt agent**: a `gpt-4.1-mini` agent rewrites the detector prompt using structured grader feedback and the failing code snippet as context.

### `evaluate.py` — Before/after comparison

Runs the **initial v1 prompt** and the **best evolved prompt** on the same test rows (never seen during training or validation) and produces a per-grader metrics table plus a raw JSON file in `results/`.

---

## Evaluation Framework

Four graders are registered as an OpenAI Eval in `data/setup_eval.py`:

| Grader | Type | Pass threshold | What it checks |
|---|---|---|---|
| `cwe_coverage_grader` | Python (deterministic) | ≥ 0.85 | Fraction of known CWEs that appear in the agent output |
| `false_positive_grader` | Python (deterministic) | ≥ 0.80 | Whether the agent incorrectly alarms on safe (patched) code |
| `cvss_accuracy_grader` | Python (deterministic) | ≥ 0.80 | Whether the predicted CVSS score falls within the NVD range |
| `remediation_quality_judge` | LLM-as-judge (`gpt-4.1`) | ≥ 0.80 | Whether the fix addresses root cause, anchored to the real CVE patch |

A row is considered a **lenient pass** if ≥ 75% of graders passed **or** the average score is ≥ 0.85.

---

## Dataset

The dataset is built from [CIRCL/vulnerability-cwe-patch](https://huggingface.co/datasets/CIRCL/vulnerability-cwe-patch) (CVEfixes) on HuggingFace, enriched with CVSS v3.1 scores from the [NVD API](https://nvd.nist.gov/developers/vulnerabilities).

### Schema

| Field | Type | Description |
|---|---|---|
| `code` | `str` | Vulnerable function or diff snippet |
| `safe_fix` | `str` | Verified real-world patch from CVEfixes |
| `language` | `str` | Detected from file extension (`python`, `c`, `java`, …) |
| `known_cwes` | `list[str]` | e.g. `["CWE-89", "CWE-79"]` |
| `cve_id` | `str` | e.g. `CVE-2023-1234` |
| `expected_severity` | `list[float]` | `[low, high]` CVSS v3.1 range from NVD |
| `is_vulnerable` | `bool` | `True` = vulnerable, `False` = safe variant |
| `source` | `str` | Dataset origin tag |

For every vulnerable example, the corresponding patched version is added as a negative (safe) example — the agent must correctly declare the patched code clean.

### Quality filters

Rows are dropped if:
- No CWE label on a vulnerable row
- Before and after code are identical (no real diff)
- Any code line exceeds 500 characters (minified/binary)
- Code is shorter than 80 characters or longer than 8,000 characters

### Train / val / test split

70% train · 15% validation · 15% test, stratified by `is_vulnerable`.

---

## Configuration

Edit the `CONFIG` dict at the top of `data/build_security_dataset.py` to control dataset size:

```python
CONFIG = {
    "MAX_VULN_ROWS": 500,       # cap on vulnerable examples
    "MAX_SAFE_ROWS": 500,       # cap on safe (patched) examples
    "LANGUAGES": None,          # None = all; or e.g. ["python", "javascript"]
    "NVD_DELAY_S": 6,           # seconds between NVD requests (no key)
    "CVSS_TOLERANCE": 1.5,      # ± band around base score
    "MIN_CODE_CHARS": 80,
    "MAX_CODE_CHARS": 8000,
}
```

Edit `evolving_loop.py` to change training behaviour:

```python
FAILURE_WINDOW = 3          # consecutive failures before prompt evolution
VAL_ROWS_FOR_SELECTION = 15 # rows used for validation-based prompt selection
```

---

## Requirements

- Python 3.12+
- OpenAI API key (for LLM calls and evals)
- NVD API key (optional, for faster CVSS enrichment)

```
openai>=2.29.0
openai-agents>=0.13.0
pandas>=3.0.1
datasets>=4.8.4
requests>=2.32.5
tqdm>=4.67.3
python-dotenv>=1.2.2
```


## References
[OpenAI Cookbook: Self-Evolving Agents](https://developers.openai.com/cookbook/examples/partners/self_evolving_agents/autonomous_agent_retraining#3-self-evolving-loop-with-llm-as-a-judge)
