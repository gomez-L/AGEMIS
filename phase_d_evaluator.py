import json
from langchain_core.output_parsers import JsonOutputParser
import main as agemis
from main import FALKOR_ENABLED
from phase_d.prompts import get_evaluator_prompt
from phase_d.simulation import simulate_policy_cypher

# Initialize the evaluator prompt
evaluator_prompt = get_evaluator_prompt(agemis.Evaluation)

def evaluate_node(state: agemis.AgentState):
    print("\n" + "="*50)
    print("--- [Phase D] EVALUATOR NODE (Policy Simulator) ---")
    
    chain = evaluator_prompt | agemis.llm | agemis.RobustJsonOutputParser(pydantic_object=agemis.Evaluation)
    evaluation = chain.invoke({"scenario": str(state["scenario"])})
    
    print(f"\nOUTPUT (evaluation): {json.dumps(evaluation, indent=2)}")
    
    cypher_sim = evaluation.get("policy_cypher_simulation")
    if cypher_sim:
        print(f"\n⚡ [EXECUTING POLICY PERTURBATION IN GRAPH]")
        print(f"  Query: {cypher_sim}")
        if FALKOR_ENABLED:
            simulate_policy_cypher(cypher_sim)
        else:
            print("  FalkorDB disabled. Simulation skipped.")
            
    print("="*50)
    return {"evaluation": evaluation}
