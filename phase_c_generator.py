import main as agemis
from phase_c.generator import generate_scenarios, scenario_digest


def _parser(pydantic_object):
    return agemis.RobustJsonOutputParser(pydantic_object=pydantic_object)


def generate_scenario_node(state: agemis.AgentState):
    print("\n" + "=" * 50)
    print("--- [Phase C] SCENARIO GENERATOR ---")

    horizon_map = state.get("horizon_map") or {}
    modules = state.get("approved_modules") or state.get("modules") or []

    scenarios, info = generate_scenarios(horizon_map, modules,
                                         llm=agemis.llm, parser_factory=_parser,
                                         weak_signals=state.get("weak_signals"))

    if not scenarios:
        print(f"  [Phase C] No scenarios: {info.get('error', 'no vulnerability seeds found')}.")
        return {"scenario": None, "scenarios": []}

    print(f"  [Phase C] World: {info['world']['modules']} modules / "
          f"{info['world']['edges']} edges; {info['seeds_considered']} vulnerability "
          f"seeds considered; {len(scenarios)} scenario(s) generated.")
    for scenario in scenarios:
        print(f"    - {scenario_digest(scenario)}")

    return {"scenario": scenarios[0], "scenarios": scenarios}
