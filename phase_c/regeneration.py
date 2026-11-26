"""Run only Phase C against inputs. State of previous phases is preserved."""

import json
import os
import tempfile
from pathlib import Path

def regenerate_saved_scenarios(state_path, generate):
    state_path = Path(state_path)
    original = state_path.read_text(encoding="utf-8")
    state = json.loads(original)
    if not state.get("modules"):
        raise ValueError("No saved modules yet. Run Phase A has to be run before.")
    if state.get("stale") or not (state.get("horizon_map") or {}).get("knowledge_graph_edges"):
        raise ValueError("No current saved horizon graph. Run Phase A-B has to be run before.")

    result = generate({"modules": state["modules"],
                       "approved_modules": state["modules"],
                       "horizon_map": state["horizon_map"],
                       "weak_signals": state.get("weak_signals") or []})
    if not result.get("scenarios"):
        raise ValueError("No scenarios generated. Saved state was left unchanged.")
    state.update(scenario=result["scenarios"][0], scenarios=result["scenarios"], evaluation=None)

    if state_path.read_text(encoding="utf-8") != original:
        raise ValueError("Saved inputs changed during generation. Retry using the updated graph.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=state_path.parent,
                                         prefix=".scenarios-", suffix=".json", delete=False) as output:
            temporary = Path(output.name)
            json.dump(state, output, indent=2, ensure_ascii=False)
            output.write("\n")
        os.replace(temporary, state_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(result["scenarios"])
