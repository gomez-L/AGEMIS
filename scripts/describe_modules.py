"""Add plain-language summaries to the modules saved in main_state.json (only missing or outdated ones).

    python scripts/describe_modules.py          # fill missing / outdated summaries
    python scripts/describe_modules.py --force  # rewrite all summaries
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from phase_a.describe import add_module_summaries

load_dotenv(ROOT / ".env")
llm = ChatOpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"), model_name=os.getenv("MODEL_NAME"),
                 temperature=0, timeout=120, max_retries=2)

state_path = ROOT / "main_state.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
modules = list(state.get("modules") or []) + list(state.get("proto_modules") or [])
written = add_module_summaries(modules, llm, force="--force" in sys.argv)
state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
print(f"Wrote {written} summaries; {sum(1 for m in modules if m.get('summary'))}/{len(modules)} modules have one.")
