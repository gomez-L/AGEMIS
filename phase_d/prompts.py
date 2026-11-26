from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

EVALUATOR_TEMPLATE = """You are the Evaluator Agent (Policy Simulator).
Evaluate the following future scenario for systemic significance and recommend policy actions.
Start with the scenario narration, supporting_analysis, watch_for, key_uncertainty and analyst_question. Use a legacy financial_stability brief when present. Respect monitoring_only and insufficient_evidence assessments: don't equate a graph cascade with systemic disruption. Distinguish supplied evidence, simulation and assumptions. Assess impaired financial functions, exposures, amplification, participant responses, containment and evidence gaps. Policy options must remain conditional on the identified transmission mechanism and constraints. Graph stress scores are not probabilities or financial loss estimates.

EVALUATION CRITERIA:
1. URGENCY: Does the scenario stem from 'Horizon Scanning Signals' requiring immediate action?
2. POLICY RELEVANCE: As a policy maker in finance, focus on preventing systemic collapse.
3. GRAPH PERTURBATION SIMULATION: To implement your `policy_action`, you must provide a valid Cypher query in `policy_cypher_simulation` that mathematically applies the policy to the FalkorDB graph.
   - SCHEMA CONSTRAINT: You MUST only use the label `(:Module)` for nodes and `[:INTERACTS]` for edges. Do not invent new labels! Nodes have properties like `name`, `velocity`, `inertia`, `impact_potential`. Edges have properties like `type`, `empirical_trend`, `institutional`.
   - SYNTAX CONSTRAINT: If your query has multiple disjoint MATCH/SET/DELETE clauses, you MUST separate them with `WITH *`. You cannot start a new `MATCH` immediately after a `SET`.
   - Example 1 (Sever a risky connection): MATCH (a:Module)-[r:INTERACTS]-(b:Module) WHERE r.type CONTAINS 'Liquidity' DELETE r
   - Example 2 (Multiple updates): MATCH (m:Module) WHERE m.impact_potential = 'High' SET m.velocity = 'Reduced' WITH * MATCH (n:Module) SET n.inertia = 'High'

Scenario: {scenario}
{format_instructions}"""

def get_evaluator_prompt(evaluation_class):
    return PromptTemplate(
        template=EVALUATOR_TEMPLATE,
        input_variables=["scenario"],
        partial_variables={"format_instructions": JsonOutputParser(pydantic_object=evaluation_class).get_format_instructions()},
    )
