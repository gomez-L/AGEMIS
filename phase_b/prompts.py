from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

MAPPER_TEMPLATE = """You are the Mapper Agent. 
Your task is to construct the current combinatorial horizon graph (knowledge graph) based on HITL-approved modules.

CRITICAL INSTRUCTIONS FOR HIGH-FIDELITY TOPOLOGY:
1. HORIZON GRAPH CONSTRUCTION (SYSTEMIC DENSITY):
   - Systemic risk is highly interconnected. Avoid creating isolated two-node pairs or leaving a large number of isolated (grey) modules.
   - You MUST identify and map a dense, cohesive web of relationships. Aim to output at least 18 to 25 distinct, valid cross-domain edges.
   - Actively search for and map multi-step causal cascades (e.g., Module A affects Module B, which in turn affects Module C) and feedback loops.
   - Connect almost every major structural module to at least one or more other modules to create a unified risk landscape.
   - STRICT MODULE NAME MATCHING: For `source_module` and `target_module`, you MUST ONLY use module names that are EXACTLY matching the names of the approved modules. Do NOT summarize, shorten, or alter these names.
   - Do not add edges only because two modules are semantically similar. Each edge must express a plausible causal or systemic interaction.
2. CAUSAL DIRECTIONALITY: For each interaction, declare the causal directionality (e.g., Unidirectional Source->Target, or Bidirectional feedback loop).
3. DYNAMIC WEIGHT ALLOCATION: Tag each interaction with weights (1-10) for:
   - Empirical Trend Strength (momentum of the interaction).
   - Narrative Salience (level of market/public attention).
   - Institutional Relevance (impact on financial supervisory/regulatory mandates).
   - Prefer relationships grounded in the evidence or reasoning inside the connected modules.
   - Prioritize edges that explain the current unstable configuration: transmission, amplification, mitigation, vulnerability accumulation, or feedback.
   - Avoid generic background associations that do not change the interpretation of the present risk topology.
4. PERTURBATION SENSITIVITY: Identify specific nodes or edges in the graph that are highly sensitive to small shocks/perturbations (Module Perturbations).
5. INFLEXION POINTS: Identify specific 'weak signals' interacting in this graph from where significant inflexion futures might suddenly emerge.
6. SYSTEMIC VULNERABILITY: Formulate a central vulnerability representing an unstable present configuration with significant future implications.

Drivers (Modules): {modules}
{format_instructions}"""

def get_mapper_prompt(horizon_map_class):
    return PromptTemplate(
        template=MAPPER_TEMPLATE,
        input_variables=["modules"],
        partial_variables={"format_instructions": JsonOutputParser(pydantic_object=horizon_map_class).get_format_instructions()},
    )
