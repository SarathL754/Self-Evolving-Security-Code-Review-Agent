import re
from pathlib import Path
import pandas as pd

_DATA_DIR = Path(__file__).parent


def strip_mail_header(text: str) -> str:
    # Find first real diff line and take everything from there
    match = re.search(r'^(diff --git|@@|\-\-\-|\+\+\+)', text, re.MULTILINE)
    return text[match.start():] if match else text


df = pd.read_csv(_DATA_DIR / "security_agent_dataset.csv")
df["code"] = df["code"].apply(strip_mail_header)
df["safe_fix"] = df["safe_fix"].apply(strip_mail_header)

# Drop rows where stripping left too little content
df = df[df["code"].str.len() > 80]
df.to_csv(_DATA_DIR / "security_agent_dataset_clean.csv", index=False)
print(f"Clean rows: {len(df)}")