"""
dataset_loader.py
=================
Reads the CSV produced by build_security_dataset.py and returns rows
in the exact schema expected by the self-evolving agent graders.

Usage:
    from dataset_loader import load_agent_dataset, iter_agent_batches

    rows = load_agent_dataset("security_agent_dataset.csv")
    for batch in iter_agent_batches(rows, batch_size=10):
        for row in batch:
            # row has: code, safe_fix, language, known_cwes,
            #          cve_id, expected_severity, is_vulnerable, source
            pass
"""

import json
import random
import pandas as pd
from typing import Iterator


def load_agent_dataset(
    path: str,
    languages: list[str] | None = None,
    max_rows: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[dict]:
    """
    Load and deserialise the dataset CSV.

    Args:
        path        : Path to the CSV file.
        languages   : Optional list to filter by language, e.g. ["python", "c"].
        max_rows    : Optional cap on total rows returned.
        shuffle     : Whether to shuffle the rows before returning.
        seed        : Random seed for shuffling.

    Returns:
        List of dicts, one per row. known_cwes and expected_severity are
        native Python objects (list), not JSON strings.
    """
    df = pd.read_csv(path)

    # Deserialise JSON columns
    df["known_cwes"] = df["known_cwes"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )
    df["expected_severity"] = df["expected_severity"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else [0.0, 10.0]
    )
    df["is_vulnerable"] = df["is_vulnerable"].astype(bool)

    # Optional language filter
    if languages:
        df = df[df["language"].isin(languages)]

    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if max_rows:
        df = df.head(max_rows)

    return df.to_dict(orient="records")


def iter_agent_batches(
    rows: list[dict],
    batch_size: int = 20,
) -> Iterator[list[dict]]:
    """Yield successive batches from the dataset."""
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def train_val_test_split(
    rows: list[dict],
    train: float = 0.7,
    val: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split rows into train / validation / test sets.
    Stratified by is_vulnerable to keep label balance.
    """
    rng = random.Random(seed)

    vuln = [r for r in rows if r["is_vulnerable"]]
    safe = [r for r in rows if not r["is_vulnerable"]]

    def _split(items):
        items = items.copy()
        rng.shuffle(items)
        n = len(items)
        t1 = int(n * train)
        t2 = int(n * (train + val))
        return items[:t1], items[t1:t2], items[t2:]

    v_tr, v_val, v_te = _split(vuln)
    s_tr, s_val, s_te = _split(safe)

    def _merge_shuffle(a, b):
        combined = a + b
        rng.shuffle(combined)
        return combined

    return (
        _merge_shuffle(v_tr, s_tr),
        _merge_shuffle(v_val, s_val),
        _merge_shuffle(v_te, s_te),
    )


# ---------------------------------------------------------------------------
# Quick dataset health check — run as a script to validate your CSV
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "security_agent_dataset.csv"

    print(f"Loading {path}…")
    rows = load_agent_dataset(path, shuffle=False)
    print(f"Total rows: {len(rows)}")

    vuln = [r for r in rows if r["is_vulnerable"]]
    safe = [r for r in rows if not r["is_vulnerable"]]
    print(f"Vulnerable : {len(vuln)}")
    print(f"Safe       : {len(safe)}")

    # CWE distribution
    from collections import Counter
    all_cwes = [cwe for r in vuln for cwe in r["known_cwes"]]
    top_cwes = Counter(all_cwes).most_common(10)
    print("\nTop 10 CWEs:")
    for cwe, count in top_cwes:
        print(f"  {cwe:12s} {count}")

    # Language distribution
    langs = Counter(r["language"] for r in rows)
    print("\nLanguages:")
    for lang, count in langs.most_common():
        print(f"  {lang:15s} {count}")

    # CVSS coverage
    with_cvss = sum(
        1 for r in vuln
        if r["expected_severity"] != [0.0, 10.0]
    )
    print(f"\nCVSS resolved : {with_cvss}/{len(vuln)} vulnerable rows")

    # Spot-check a random vulnerable row
    import random
    sample = random.choice(vuln)
    print(f"\nRandom vulnerable sample:")
    print(f"  CVE              : {sample['cve_id']}")
    print(f"  CWEs             : {sample['known_cwes']}")
    print(f"  Expected severity: {sample['expected_severity']}")
    print(f"  Language         : {sample['language']}")
    print(f"  Code (200 chars) : {sample['code'][:200].strip()}")
    print(f"  Fix  (200 chars) : {sample['safe_fix'][:200].strip()}")

    # Split check
    train, val, test = train_val_test_split(rows)
    print(f"\nSplit sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}")
