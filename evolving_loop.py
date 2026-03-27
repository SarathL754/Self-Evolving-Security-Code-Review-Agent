"""
evolving_loop.py — Step 5
==========================
Self-evolving loop that:
  1. Runs the security agent pipeline on training rows.
  2. Evaluates each output with the OpenAI Eval created in setup_eval.py.
  3. If a row fails, rewrites the detector prompt via a metaprompt agent.
  4. Tracks which prompt version earned the best average score.

Usage:
    python evolving_loop.py
"""

import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "data"))
from dataset_loader import load_agent_dataset, train_val_test_split
from agents import vulnerability_detector_prompt, make_detector_agent, run_pipeline

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()
_api_key: str = os.getenv("OPENAI_API_KEY") or ""
EVAL_ID: str = os.getenv("EVAL_ID") or ""

if not _api_key:
    raise EnvironmentError("OPENAI_API_KEY not set in .env")
if not EVAL_ID:
    raise EnvironmentError("EVAL_ID not set in .env — run data/setup_eval.py first")

client = OpenAI(api_key=_api_key)

# Number of consecutive row failures required before the prompt is evolved.
# Prevents overfitting to a single example.
FAILURE_WINDOW = 3

# Placeholder patterns that indicate the metaprompt agent hallucinated a
# template stub instead of writing a real prompt.
_PLACEHOLDER_PATTERNS = ("[Insert", "[insert", "[CODE", "[Code", "<<CODE>>", "<<code>>")

# Maximum validation rows used to select the best prompt at end of training.
VAL_ROWS_FOR_SELECTION = 15


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def run_eval(
    eval_id: str,
    code: str,
    safe_fix: str,
    known_cwes,
    is_vulnerable: bool,
    expected_severity,
    output: str,
) -> str:
    """
    Submit a single-item eval run and return the run ID.

    known_cwes / expected_severity may be lists (from dataset_loader) or
    JSON strings (from the raw CSV). Both are normalised to JSON strings
    because the graders call json.loads() on them.
    """
    def _to_json_str(v):
        return json.dumps(v) if isinstance(v, list) else str(v)

    run = client.evals.runs.create(
        eval_id=eval_id,
        data_source={
            "type": "jsonl",
            "source": {
                "type": "file_content",
                "content": [
                    {
                        "item": {
                            "code": code,
                            "safe_fix": safe_fix,
                            "known_cwes": _to_json_str(known_cwes),
                            "is_vulnerable": is_vulnerable,
                            "expected_severity": _to_json_str(expected_severity),
                        },
                        "sample": {
                            "output_text": output,
                        },
                    }
                ],
            },
        },
    )
    return run.id


def poll_eval_run(eval_id: str, run_id: str, max_attempts: int = 10, sleep_s: int = 5):
    """
    Poll until the run reaches 'completed'. Raises on failure or timeout.
    Returns the completed run object.
    """
    for attempt in range(max_attempts):
        run = client.evals.runs.retrieve(eval_id=eval_id, run_id=run_id)
        if run.status == "completed":
            return run
        if run.status in ("failed", "cancelled"):
            raise RuntimeError(f"Eval run {run_id} ended with status '{run.status}'")
        time.sleep(sleep_s)
    raise TimeoutError(
        f"Eval run {run_id} did not complete after {max_attempts} attempts ({max_attempts * sleep_s}s)"
    )


def parse_eval_run_output(items) -> list[dict]:
    """
    Extract a flat list of grader results from output_items.
    Each entry: {"name", "score", "passed", "reasoning"}
    """
    grader_scores = []
    for item in items:
        for result in getattr(item, "results", []) or []:
            grader_scores.append({
                "name": getattr(result, "name", "unknown"),
                "score": float(getattr(result, "score", 0.0) or 0.0),
                "passed": bool(getattr(result, "passed", False)),
                "reasoning": (
                    getattr(result, "reasoning", None)
                    or getattr(result, "reason", None)
                    or ""
                ),
            })
    return grader_scores


def calculate_grader_score(grader_scores: list[dict]) -> float:
    """Average score across all graders. Returns 0.0 for an empty list."""
    if not grader_scores:
        return 0.0
    return sum(g["score"] for g in grader_scores) / len(grader_scores)


def is_lenient_pass(grader_scores: list[dict], average_score: float) -> bool:
    """True if ≥75 % of graders passed OR average score ≥ 0.85."""
    if not grader_scores:
        return False
    pass_rate = sum(1 for g in grader_scores if g["passed"]) / len(grader_scores)
    return pass_rate >= 0.75 or average_score >= 0.85


def collect_security_feedback(grader_scores: list[dict]) -> str:
    """
    Plain-English description of which graders failed and why.
    Domain-specific messages tied to each grader name.
    """
    messages = []
    for g in grader_scores:
        if g["passed"]:
            continue
        name = g["name"]
        if name == "cwe_coverage_grader":
            messages.append(
                "Agent missed known CWE IDs. "
                "Improve prompt to systematically check OWASP Top 10 categories."
            )
        elif name == "false_positive_grader":
            messages.append(
                "Agent raised false alarm on safe code. "
                "Improve prompt to require evidence of exploitability before flagging."
            )
        elif name == "cvss_accuracy_grader":
            messages.append(
                "Agent CVSS score was off. "
                "Improve prompt to anchor severity to attack vector, complexity, and impact."
            )
        elif name == "remediation_quality_judge":
            reasoning = g.get("reasoning", "").strip()
            messages.append(
                f"Remediation quality was insufficient. Judge reasoning: {reasoning}"
            )
        else:
            messages.append(f"Grader '{name}' failed (score={g['score']:.3f}).")
    return "\n".join(messages) if messages else "All graders passed."


# ---------------------------------------------------------------------------
# Metaprompt agent
# ---------------------------------------------------------------------------

_METAPROMPT_TEMPLATE = """\
Original prompt: {original_prompt}

Code reviewed (truncated):
{code}

Expected CWEs the agent should have identified: {known_cwes}
Expected CVSS severity range: {expected_severity}

Agent output (truncated):
{agent_output}

Grader feedback (what failed and why):
{grader_feedback}

Write an improved vulnerability detection prompt that:
- References specific CWE identifiers for every finding
- Distinguishes confirmed vulnerabilities from code smells
- Never flags safe defensive code as vulnerable
- Covers OWASP Top 10 categories before declaring code clean
- States explicitly when code is safe rather than staying silent
- Instructs the agent to output a numeric CVSS v3.1 base score (0.0–10.0)
- Instructs the agent to list all CWE identifiers found in a structured section
Output only the new prompt text, nothing else. Do not include placeholders like \
[Insert code here] — the code will be appended automatically."""

_metaprompt_agent = Agent(
    name="metaprompt_agent",
    instructions=(
        "You are an expert prompt engineer specialising in security AI systems. "
        "Given an underperforming detector prompt and structured grader feedback, "
        "rewrite the prompt to be more precise, systematic, and accurate. "
        "Return only the new prompt text — no explanations, no preamble."
    ),
    model="gpt-4.1-mini",
)


# ---------------------------------------------------------------------------
# Self-evolving loop
# ---------------------------------------------------------------------------

async def _score_prompt_on_rows(
    prompt: str,
    model: str,
    rows: list[dict],
    label: str,
) -> float:
    """
    Run a prompt against a list of rows through the eval pipeline and return
    the average score. Used for validation-based prompt selection.
    """
    from agents import Agent as _Agent  # local import to avoid circular reference
    agent = _Agent(name="vulnerability_detector", instructions=prompt, model=model)
    scores = []
    for idx, row in enumerate(rows):
        print(f"  [val/{label}]  row {idx + 1:>2}/{len(rows)}", end="\r", flush=True)
        try:
            output = await run_pipeline(row["code"], agent)
            run_id = run_eval(
                EVAL_ID,
                row["code"], row["safe_fix"], row["known_cwes"],
                row["is_vulnerable"], row["expected_severity"],
                output,
            )
            poll_eval_run(EVAL_ID, run_id)
            items = list(client.evals.runs.output_items.list(eval_id=EVAL_ID, run_id=run_id))
            grader_scores = parse_eval_run_output(items)
            scores.append(calculate_grader_score(grader_scores))
        except Exception as exc:
            print(f"\n  [val/{label}]  row {idx + 1} error: {exc}")
            scores.append(0.0)
    print()
    return sum(scores) / len(scores) if scores else 0.0


async def self_evolving_loop() -> None:
    csv_path = Path(__file__).parent / "data" / "security_agent_dataset_clean.csv"

    # Load a larger training set for better generalisation
    rows = load_agent_dataset(str(csv_path), max_rows=60, shuffle=True)
    print(f"Loaded {len(rows)} rows for training.\n")

    # Maps prompt_version → scores earned while that version was active
    version_scores: dict[int, list[float]] = defaultdict(list)
    # Consecutive failed rows under the current prompt
    consecutive_failures = 0

    for row_idx, row in enumerate(rows):
        code              = row["code"]
        safe_fix          = row["safe_fix"]
        known_cwes        = row["known_cwes"]
        is_vulnerable     = row["is_vulnerable"]
        expected_severity = row["expected_severity"]

        cve = row.get("cve_id", "N/A")
        print(f"[{row_idx + 1:02d}/{len(rows)}] CVE={cve}  vulnerable={is_vulnerable}")

        current_v = vulnerability_detector_prompt.current()
        detector  = make_detector_agent()

        # 1. Run pipeline
        try:
            output = await run_pipeline(code, detector)
        except Exception as exc:
            print(f"         Pipeline error: {exc}")
            version_scores[current_v.version].append(0.0)
            consecutive_failures += 1
            continue

        # 2. Submit eval
        try:
            run_id = run_eval(
                EVAL_ID, code, safe_fix, known_cwes,
                is_vulnerable, expected_severity, output,
            )
        except Exception as exc:
            print(f"         Eval submit error: {exc}")
            version_scores[current_v.version].append(0.0)
            consecutive_failures += 1
            continue

        # 3. Poll for results
        try:
            poll_eval_run(EVAL_ID, run_id)
            output_items = list(
                client.evals.runs.output_items.list(eval_id=EVAL_ID, run_id=run_id)
            )
        except (TimeoutError, RuntimeError) as exc:
            print(f"         Eval poll error: {exc}")
            version_scores[current_v.version].append(0.0)
            consecutive_failures += 1
            continue

        # 4. Parse & score — attribute to the version that was active this row
        grader_scores = parse_eval_run_output(output_items)
        avg    = calculate_grader_score(grader_scores)
        passed = is_lenient_pass(grader_scores, avg)

        version_scores[current_v.version].append(avg)
        status = "PASS" if passed else "FAIL"
        print(f"         score={avg:.3f}  [{status}]  prompt_v{current_v.version}  "
              f"(consecutive_failures={consecutive_failures})")

        if passed:
            consecutive_failures = 0
        else:
            consecutive_failures += 1

            # 5. Evolve only after FAILURE_WINDOW consecutive failures
            if consecutive_failures >= FAILURE_WINDOW:
                feedback = collect_security_feedback(grader_scores)
                metaprompt_input = _METAPROMPT_TEMPLATE.format(
                    original_prompt=current_v.prompt,
                    code=code[:600],
                    agent_output=output[:600],
                    grader_feedback=feedback,
                    known_cwes=known_cwes,
                    expected_severity=expected_severity,
                )
                try:
                    meta_result = await Runner.run(_metaprompt_agent, metaprompt_input)
                    new_prompt  = meta_result.final_output.strip()

                    # Guard: reject prompts that contain placeholder stubs
                    if any(p in new_prompt for p in _PLACEHOLDER_PATTERNS):
                        print(
                            f"         → evolved prompt rejected (placeholder detected), "
                            f"keeping v{current_v.version}"
                        )
                    else:
                        vulnerability_detector_prompt.update(
                            new_prompt,
                            metadata={
                                "evolved_on_row": row_idx + 1,
                                "consecutive_failures": consecutive_failures,
                                "trigger_score": round(avg, 4),
                            },
                        )
                        new_v = vulnerability_detector_prompt.current().version
                        print(f"         → prompt evolved v{current_v.version} → v{new_v}")
                        consecutive_failures = 0
                except Exception as exc:
                    print(f"         Metaprompt error: {exc}")

    # ── Training-score summary ────────────────────────────────────────────────
    train_averages = {
        v: sum(scores) / len(scores)
        for v, scores in version_scores.items()
        if scores
    }

    print("\n" + "=" * 64)
    print("TRAINING COMPLETE — per-version training scores:")
    for v in sorted(train_averages):
        n = len(version_scores[v])
        print(f"  v{v}: avg={train_averages[v]:.4f}  over {n} row(s)")

    # ── Validation-based best prompt selection ────────────────────────────────
    # Score every candidate version on a held-out validation split so that the
    # best prompt is chosen on unseen data, not the rows it was optimised on.
    print(f"\nScoring {len(vulnerability_detector_prompt.history)} prompt version(s) "
          f"on up to {VAL_ROWS_FOR_SELECTION} validation rows…")

    all_rows_for_split = load_agent_dataset(str(csv_path))
    _, val_rows, _ = train_val_test_split(all_rows_for_split)
    val_rows = val_rows[:VAL_ROWS_FOR_SELECTION]

    val_averages: dict[int, float] = {}
    for pv in vulnerability_detector_prompt.history:
        val_avg = await _score_prompt_on_rows(pv.prompt, pv.model, val_rows, f"v{pv.version}")
        val_averages[pv.version] = val_avg
        print(f"  v{pv.version} val_avg={val_avg:.4f}")

    best_version = max(val_averages, key=val_averages.__getitem__)
    best_avg_val = val_averages[best_version]

    best_prompt_obj = next(
        (pv for pv in vulnerability_detector_prompt.history if pv.version == best_version),
        vulnerability_detector_prompt.current(),
    )

    final_version = vulnerability_detector_prompt.current().version

    print("\n" + "=" * 64)
    print("SELF-EVOLVING LOOP COMPLETE")
    print("=" * 64)
    print(f"Final prompt version : v{final_version}")
    print(f"Best prompt version  : v{best_version}  (val avg score {best_avg_val:.4f})")
    print(f"\nBest prompt text (v{best_version}):\n")
    print(best_prompt_obj.prompt)
    print("=" * 64)

    # ── Persist best prompt for evaluate.py ──────────────────────────────────
    save_path = Path(__file__).parent / "best_prompt.json"
    save_path.write_text(json.dumps({
        "version": best_version,
        "prompt": best_prompt_obj.prompt,
        "model": best_prompt_obj.model,
        "avg_train_score": round(train_averages.get(best_version, 0.0), 4),
        "avg_val_score": round(best_avg_val, 4),
        "timestamp": best_prompt_obj.timestamp,
        "metadata": best_prompt_obj.metadata,
        "all_versions": [
            {
                "version": pv.version,
                "prompt": pv.prompt,
                "model": pv.model,
                "timestamp": pv.timestamp,
                "avg_train_score": round(train_averages.get(pv.version, 0.0), 4),
                "avg_val_score": round(val_averages.get(pv.version, 0.0), 4),
                "metadata": pv.metadata,
            }
            for pv in vulnerability_detector_prompt.history
        ],
    }, indent=2))
    print(f"\nBest prompt saved → {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(self_evolving_loop())
