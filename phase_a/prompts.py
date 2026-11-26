from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from phase_a.utils import load_domains

DECOMPOSER_TEMPLATE = """You are the Synthesis Engine.
Your task is to integrate a new financial/geopolitical event into a set of overarching 3D 'Modules' (structural subsystems).

CURRENT 3D MODULES:
{current_modules}

NEW EVENT:
{event}

THINKING GUIDELINES:
1. REUSE STRUCTURAL MODULES BEFORE CREATING:
   - A module represents a recurring structural mechanism that can accumulate evidence from different events. Separate aspects of one report do not automatically require separate modules.
   - First compare the event's mechanism with the closest CURRENT 3D MODULES. Use their reasoning, domains, horizon and evidence titles to understand their scope; a different headline, geography, institution or application alone is not a new mechanism.
   - UPDATE an existing module when the event supports or changes its underlying mechanism, even if the wording differs. Use its exact name and explain the fit in the update reasoning. Multiple updates are appropriate only when the event supports genuinely distinct mechanisms.
   - CREATE only when a distinct structural mechanism cannot reasonably fit the retrieved modules. In the new module's `reasoning`, identify the closest existing module(s) by name and explain the specific difference in scope or mechanism that makes a update insufficient. If none are relevant or the list is empty it should be stated explicitly.
   - Use a concise, reusable mechanism name for a creation. Keep narrower aspects in its reasoning and evidence instead of creating a module for each aspect. For example, a report on several cyberattack techniques may support one shared operational-disruption mechanism; distinct transmission mechanisms may still justify separate modules.
   - A genuinely new emerging mechanism may start with one signal; do not force it into an unrelated module. Do not create several modules with substantially overlapping scope from the same event. Attach only evidence actually supplied by the event.
   - MERGE only when the event shows that two existing modules represent the same structural subsystem, rather than merely interacting. Return a `MergeModuleAction` with the consolidated module and rational.
   - DISCARD an event only when it is noise or irrelevant to systemic risk, with a reason. Do not discard a relevant event just to reduce the module count.

2. 3D MODULE ATTRIBUTES (if updating or creating):
   - Velocity: How fast is this structural subsystem evolving now?
   - Inertia: What is the resistance or structural momentum of this subsystem?
   - STEEPS Intersections: Which of the following configured domains intersect within this module?
     - {domains_text}

3. NOVELTY:
   - Determine the novelty of the module based purely on your own internal knowledge and the current signals. Is this a common historical trend, or something fundamentally new and disruptive?
   - Update `novelty_score` (1-10) and `novelty_rationale` accordingly.

Return ONLY the explicit actions (`updates`, `creations`, `merges`, `discards`) using the schema below. Use empty lists for actions that are not needed. Do not return modules that require no changes.

{format_instructions}"""

def get_decomposer_prompt(synthesis_result_class):
    return PromptTemplate(
        template=DECOMPOSER_TEMPLATE,
        input_variables=["current_modules", "event"],
        partial_variables={
            "format_instructions": JsonOutputParser(pydantic_object=synthesis_result_class).get_format_instructions(),
            "domains_text": load_domains()
        },
    )
