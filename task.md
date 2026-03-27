Project: Self-evolving security code review agent using the OpenAI Agents SDK.
What's already built:

data/security_agent_dataset_clean.csv — 1,000 rows (500 vulnerable, 500 safe) with columns: code, safe_fix, language, known_cwes, expected_severity, is_vulnerable, source
data/dataset_loader.py — loads the CSV, deserialises JSON columns, provides train_val_test_split()
data/setup_eval.py — already run, created the OpenAI eval. EVAL_ID is saved in .env
.env — contains OPENAI_API_KEY and EVAL_ID

What needs to be built now (Steps 4 and 5):

Step 4 — Three sub-agents in agents.py
Create a file agents.py at the project root with the following:
A VersionedPrompt class that tracks prompt history. Each version stores: version (int), prompt (str), model (str), timestamp, and optional metadata dict. Must support .current(), .update(new_prompt, model, metadata), and .revert_to_version(n).
Three agents using the OpenAI Agents SDK (from agents import Agent):
The first is the vulnerability_detector — this is the agent that evolves. Start it with a deliberately weak prompt: "You are a security code reviewer. Given a code snippet, identify any security vulnerabilities." Wrap it in VersionedPrompt so its prompt can be updated by the loop. Model: gpt-4.1-mini.
The second is the severity_classifier — static prompt, never evolves. Prompt: "You are a CVSS scoring assistant. Given a vulnerability description, assign a CVSS v3.1 base score. Output the score as a decimal number (e.g. 7.5) and one sentence of justification." Model: gpt-4.1-mini.
The third is the remediation_advisor — static prompt, never evolves. Prompt: "You are a secure coding expert. Given a vulnerability and the original code, provide: 1) The CWE identifier, 2) A corrected version of the code, 3) One sentence explaining why the fix works. Never suggest string sanitization for injection flaws. Always recommend parameterized queries for SQL. Always recommend allowlist validation for path traversal." Model: gpt-4.1-mini.
Also create a run_pipeline(code: str, detector_agent) -> str async function that runs all three agents in sequence — detector output feeds into classifier, classifier output feeds into advisor — and returns the combined output as a single string.

Step 5 — Self-evolving loop in evolving_loop.py
Create evolving_loop.py at the project root. It imports from agents.py, data/dataset_loader.py, and reuses the eval helpers from the cookbook pattern.
Needs these helpers (same as cookbook, adapted for security):
run_eval(eval_id, code, safe_fix, known_cwes, is_vulnerable, expected_severity, output) — calls client.evals.runs.create with the item fields and agent output.
poll_eval_run(eval_id, run_id) — polls until status is completed, max 10 attempts, 5s sleep between attempts.
parse_eval_run_output(items) — extracts grader name, score, passed, and reasoning from each result.
calculate_grader_score(grader_scores) — returns average score across all graders.
is_lenient_pass(grader_scores, average_score) — returns True if 75% of graders passed OR average ≥ 0.85.
collect_security_feedback(grader_scores) — returns a plain English string describing which graders failed and why. Domain-specific messages:

cwe_coverage_grader failed → "Agent missed known CWE IDs. Improve prompt to systematically check OWASP Top 10 categories."
false_positive_grader failed → "Agent raised false alarm on safe code. Improve prompt to require evidence of exploitability before flagging."
cvss_accuracy_grader failed → "Agent CVSS score was off. Improve prompt to anchor severity to attack vector, complexity, and impact."
remediation_quality_judge failed → include the judge's reasoning text

The metaprompt template (passed to the metaprompt agent when graders fail):
Original prompt: {original_prompt}
Code reviewed: {code}
Agent output: {agent_output}
Grader feedback: {grader_feedback}

Write an improved vulnerability detection prompt that:
- References specific CWE identifiers for every finding
- Distinguishes confirmed vulnerabilities from code smells
- Never flags safe defensive code as vulnerable
- Covers OWASP Top 10 categories before declaring code clean
- States explicitly when code is safe rather than staying silent
Output only the new prompt, nothing else.
The main self_evolving_loop() async function should:

Load training rows from data/security_agent_dataset_clean.csv using dataset_loader.py
For each row (up to 50 rows to start): run run_pipeline(), evaluate with run_eval(), check is_lenient_pass(). If it fails, collect feedback, run the metaprompt agent to rewrite the detector prompt, update VersionedPrompt, retry up to 3 times.
Track the best prompt version by average score across all rows.
At the end print: final prompt version, best average score, and the full text of the best prompt.

A __main__ block that calls asyncio.run(self_evolving_loop()).

Environment:

pip install openai openai-agents python-dotenv pandas
All OpenAI API calls use OPENAI_API_KEY from .env
EVAL_ID loaded from .env
Use from agents import Agent, Runner, trace for the SDK

Do not build a UI. Do not add LangGraph. Keep it to two files: agents.py and evolving_loop.py.