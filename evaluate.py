"""
evaluate.py
===========
Before-vs-after evaluation on the held-out test split.

Runs the full pipeline TWICE on the same unseen rows:
  • BEFORE — using the initial weak prompt (v1)
  • AFTER  — using the best evolved prompt (vN, from best_prompt.json)

Then prints a side-by-side metrics table and saves raw results to JSON.

Prerequisites:
    1. pip install openai openai-agents python-dotenv pandas
    2. python evolving_loop.py        ← produces best_prompt.json

Usage:
    python evaluate.py                # 30 test rows (default)
    python evaluate.py --rows 50      # use more rows
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Reuse infrastructure from evolving_loop (client, EVAL_ID, helpers) ───────
from evolving_loop import (
    EVAL_ID,
    client,
    run_eval,
    poll_eval_run,
    parse_eval_run_output,
    calculate_grader_score,
    is_lenient_pass,
)

# ── Dataset + pipeline ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "data"))
from dataset_loader import load_agent_dataset, train_val_test_split
from agents import run_pipeline, vulnerability_detector_prompt

# The `agents` module in this project shadows the openai-agents package name,
# so we import Agent/Runner through our agents.py (which re-exports them from SDK).
from agents import Agent  # re-exported by our agents.py bootstrap


# ---------------------------------------------------------------------------
# Grader display names and pass thresholds (mirrors setup_eval.py)
# ---------------------------------------------------------------------------
GRADER_META = {
    "cwe_coverage_grader":       {"label": "CWE Coverage",          "threshold": 0.85},
    "false_positive_grader":     {"label": "False Positive Penalty", "threshold": 0.80},
    "cvss_accuracy_grader":      {"label": "CVSS Accuracy",          "threshold": 0.80},
    "remediation_quality_judge": {"label": "Remediation Quality",    "threshold": 0.80},
}
GRADER_ORDER = list(GRADER_META.keys())


# ---------------------------------------------------------------------------
# Core evaluation runner
# ---------------------------------------------------------------------------

async def eval_prompt(
    prompt: str,
    model: str,
    rows: list[dict],
    label: str,
) -> dict:
    """
    Run every row through run_pipeline + OpenAI Eval and aggregate metrics.

    Returns:
        {
          "avg_score":  float,
          "pass_rate":  float,          # fraction of rows passing is_lenient_pass
          "graders": {
              grader_name: {"avg_score": float, "pass_rate": float, "n": int}
          },
          "row_scores": [float, ...],   # one per row
          "n_rows":     int,
        }
    """
    agent = Agent(name="vulnerability_detector", instructions=prompt, model=model)

    row_scores: list[float] = []
    row_passed: list[bool]  = []
    grader_data: dict[str, list[dict]] = defaultdict(list)  # name → [{score, passed}]

    n = len(rows)
    for idx, row in enumerate(rows):
        print(f"  [{label}]  row {idx + 1:>3}/{n}", end="\r", flush=True)

        try:
            output = await run_pipeline(row["code"], agent)

            run_id = run_eval(
                EVAL_ID,
                row["code"], row["safe_fix"], row["known_cwes"],
                row["is_vulnerable"], row["expected_severity"],
                output,
            )
            poll_eval_run(EVAL_ID, run_id)
            items = list(client.evals.runs.output_items.list(
                eval_id=EVAL_ID, run_id=run_id
            ))
            grader_scores = parse_eval_run_output(items)

            avg   = calculate_grader_score(grader_scores)
            passed = is_lenient_pass(grader_scores, avg)

            row_scores.append(avg)
            row_passed.append(passed)

            for g in grader_scores:
                grader_data[g["name"]].append({
                    "score":  g["score"],
                    "passed": g["passed"],
                })

        except Exception as exc:
            print(f"\n  [!] row {idx + 1} error: {exc}")
            row_scores.append(0.0)
            row_passed.append(False)

    print()  # clear the \r line

    grader_summary = {}
    for name, results in grader_data.items():
        n_g = len(results)
        grader_summary[name] = {
            "avg_score": round(sum(r["score"]  for r in results) / n_g, 4),
            "pass_rate": round(sum(r["passed"] for r in results) / n_g, 4),
            "n": n_g,
        }

    return {
        "avg_score": round(sum(row_scores) / len(row_scores), 4) if row_scores else 0.0,
        "pass_rate": round(sum(row_passed) / len(row_passed), 4) if row_passed else 0.0,
        "graders":   grader_summary,
        "row_scores": row_scores,
        "n_rows":    len(rows),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(val: float) -> str:
    return f"{val * 100:.1f} %"

def _delta_str(before: float, after: float) -> str:
    d = after - before
    sign = "+" if d >= 0 else ""
    pct = (d / before * 100) if before > 0 else 0.0
    pct_sign = "+" if pct >= 0 else ""
    return f"{sign}{d:.4f}  ({pct_sign}{pct:.1f}%)"

def _delta_pp(before: float, after: float) -> str:
    """Delta in percentage points."""
    pp = (after - before) * 100
    sign = "+" if pp >= 0 else ""
    return f"{sign}{pp:.1f} pp"


def print_report(
    before: dict,
    after: dict,
    vN_version: int,
    n_rows: int,
) -> None:
    W = 76
    col_w = 14

    def row(label: str, b: str, a: str, d: str, indent: int = 0) -> str:
        pad = "  " * indent
        return f"  {pad}{label:<30}{b:>{col_w}}{a:>{col_w}}  {d}"

    sep   = "=" * W
    thin  = "-" * W

    print()
    print(sep)
    print(f"  SECURITY AGENT — BEFORE vs AFTER EVALUATION")
    print(f"  {n_rows} held-out test rows  |  never seen during training")
    print(sep)
    print(f"  {'Metric':<30}{'BEFORE (v1)':>{col_w}}{'AFTER (v' + str(vN_version) + ')':>{col_w}}  DELTA")
    print(thin)

    # ── Overall ──────────────────────────────────────────────────────────────
    print(row(
        "Overall avg score",
        f"{before['avg_score']:.4f}",
        f"{after['avg_score']:.4f}",
        _delta_str(before["avg_score"], after["avg_score"]),
    ))
    print(row(
        "Pass rate (lenient)",
        _pct(before["pass_rate"]),
        _pct(after["pass_rate"]),
        _delta_pp(before["pass_rate"], after["pass_rate"]),
    ))
    print(thin)

    # ── Per-grader ────────────────────────────────────────────────────────────
    print(f"  {'Grader':<30}{'avg  pass%':>{col_w}}{'avg  pass%':>{col_w}}  avg delta")
    print(thin)

    for grader_name in GRADER_ORDER:
        display = GRADER_META.get(grader_name, {}).get("label", grader_name)
        b_g = before["graders"].get(grader_name, {})
        a_g = after["graders"].get(grader_name, {})

        b_avg  = b_g.get("avg_score", 0.0)
        a_avg  = a_g.get("avg_score", 0.0)
        b_pass = b_g.get("pass_rate", 0.0)
        a_pass = a_g.get("pass_rate", 0.0)

        b_str = f"{b_avg:.3f}  {b_pass * 100:4.0f}%"
        a_str = f"{a_avg:.3f}  {a_pass * 100:4.0f}%"

        d = a_avg - b_avg
        sign = "+" if d >= 0 else ""
        d_str = f"{sign}{d:.4f}"

        print(f"  {display:<30}{b_str:>{col_w}}{a_str:>{col_w}}  {d_str}")

    print(sep)


def save_results(
    before: dict,
    after: dict,
    v1_prompt: str,
    best_prompt_data: dict,
    n_rows: int,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = {
        "timestamp": ts,
        "n_rows": n_rows,
        "eval_id": EVAL_ID,
        "before": {
            "version": 1,
            "prompt": v1_prompt,
            "avg_score": before["avg_score"],
            "pass_rate": before["pass_rate"],
            "graders":   before["graders"],
            "row_scores": before["row_scores"],
        },
        "after": {
            "version": best_prompt_data["version"],
            "prompt":  best_prompt_data["prompt"],
            "model":   best_prompt_data["model"],
            "avg_score": after["avg_score"],
            "pass_rate": after["pass_rate"],
            "graders":   after["graders"],
            "row_scores": after["row_scores"],
        },
        "delta": {
            "avg_score": round(after["avg_score"] - before["avg_score"], 4),
            "pass_rate": round(after["pass_rate"] - before["pass_rate"], 4),
            "graders": {
                name: round(
                    after["graders"].get(name, {}).get("avg_score", 0.0)
                    - before["graders"].get(name, {}).get("avg_score", 0.0),
                    4,
                )
                for name in GRADER_ORDER
            },
        },
    }
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    path = results_dir / f"eval_results_{ts}.json"
    path.write_text(json.dumps(out, indent=2))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(n_rows: int) -> None:
    # ── Sanity checks ─────────────────────────────────────────────────────────
    best_prompt_file = Path(__file__).parent / "best_prompt.json"
    if not best_prompt_file.exists():
        print(
            "\nERROR: best_prompt.json not found.\n"
            "Run  python evolving_loop.py  first to evolve and save the best prompt.\n"
        )
        sys.exit(1)

    best_prompt_data = json.loads(best_prompt_file.read_text())
    vN_prompt  = best_prompt_data["prompt"]
    vN_version = best_prompt_data["version"]
    vN_model   = best_prompt_data["model"]

    # v1 = the initial weak prompt (history[0] of the freshly imported VersionedPrompt)
    v1_obj    = vulnerability_detector_prompt.history[0]
    v1_prompt = v1_obj.prompt
    v1_model  = v1_obj.model

    if vN_version == 1:
        print("\nNOTE: best_prompt.json shows version 1 (no evolution occurred).")
        print("      Before and After results will be identical.\n")

    # ── Load test split ───────────────────────────────────────────────────────
    csv_path = Path(__file__).parent / "data" / "security_agent_dataset_clean.csv"
    all_rows = load_agent_dataset(str(csv_path), shuffle=True)
    _, _, test_rows = train_val_test_split(all_rows)
    test_rows = test_rows[:n_rows]

    print(f"\nTest split total: {len(test_rows)} rows  (capped to {n_rows})")
    print(f"Evaluating  v1  (initial prompt)  vs  v{vN_version}  (best evolved prompt)\n")

    # ── BEFORE ────────────────────────────────────────────────────────────────
    print(f"--- BEFORE (v1) {'─' * 48}")
    before = await eval_prompt(v1_prompt, v1_model, test_rows, label="BEFORE v1")

    # ── AFTER ─────────────────────────────────────────────────────────────────
    print(f"--- AFTER  (v{vN_version}) {'─' * 47}")
    after = await eval_prompt(vN_prompt, vN_model, test_rows, label=f"AFTER  v{vN_version}")

    # ── Report ────────────────────────────────────────────────────────────────
    print_report(before, after, vN_version=vN_version, n_rows=n_rows)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = save_results(before, after, v1_prompt, best_prompt_data, n_rows)
    print(f"  Raw results saved → {out_path.name}")
    print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Before-vs-after eval on test split")
    parser.add_argument(
        "--rows", type=int, default=50,
        help="Number of test rows to evaluate (default: 50)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.rows))
