import json
from datetime import datetime, timezone
from langchain_core.output_parsers import JsonOutputParser
import main as agemis
from main import FALKOR_ENABLED
from phase_a.utils import module_prompt_context, safe_get_module
from phase_a.prompts import get_decomposer_prompt
from phase_a.graphrag import graphrag_retrieval, persist_falkordb_actions

# Initialize the decomposer prompt
decomposer_prompt = get_decomposer_prompt(agemis.SynthesisResult)

def decompose_node(state: agemis.AgentState):
    print("\n" + "="*50)
    print("--- [Phase A] DECOMPOSE NODE (GraphRAG Synthesis) ---")
    
    events = state.get("input_events", [])
    active_modules = set()
    # HITL two-gate support: with defer_persistence, db writes are collected and returned instead of executed => so a reviewer can reject the synthesis without any store writes
    deferred_persistence = []

    # Maintain running list of 3D modules in a dict
    current_modules_dict = {m['name']: m for m in (state.get("modules") or [])}
        
    list_parser = JsonOutputParser(pydantic_object=agemis.SynthesisResult)
    chain = decomposer_prompt | agemis.llm | list_parser
    
    for idx, event in enumerate(events):
        print(f"\nIntegrating Event {idx+1}/{len(events)}: {event.get('title', 'Unknown')}")
        
        # GraphRAG: Retrieve specific context of existing modules
        event_text_for_embedding = f"{event.get('title', '')}\n{event.get('content', '')}"
        retrieved_modules = graphrag_retrieval(event_text_for_embedding, current_modules_dict)
        context_modules = [module_prompt_context(m) for m in retrieved_modules]
        
        try:
            response = chain.invoke({
                "current_modules": json.dumps(context_modules, indent=2) if context_modules else "[]",
                "event": json.dumps(event, indent=2)
            })
            
            if FALKOR_ENABLED:
                if state.get("defer_persistence"):
                    deferred_persistence.append((response, event))
                else:
                    persist_falkordb_actions(response, event)
            
            # Apply discards
            for discard in (response.get("discards") or []):
                print(f"  [DISCARD] {discard.get('event_title', 'Unknown')} | Reason: {discard.get('reasoning')}")

            # Apply updates
            for update in response.get("updates", []):
                m_name = safe_get_module(update["module_name"], current_modules_dict)
                if m_name:
                    active_modules.add(m_name)
                    mod = current_modules_dict[m_name]
                    mod["velocity"] = update["updated_velocity"]
                    mod["inertia"] = update["updated_inertia"]
                    mod["impact_potential"] = update["updated_impact_potential"]
                    mod["certainty_level"] = update["updated_certainty_level"]
                    mod["reasoning"] = f"{mod.get('reasoning', '')}\nUpdate Reasoning: {update['reasoning']}".strip()
                    
                    evidence = mod.get("evidence_signals", [])
                    evidence.append({
                        "title": event.get("title", "Unknown"),
                        "content": event.get("content", "")
                    })
                    mod["evidence_signals"] = evidence
                    print(f"  [UPDATE] {m_name}")
            
            # Apply creations
            for creation in response.get("creations", []):
                m_name = creation["name"]
                active_modules.add(m_name)
                if not creation.get("evidence_signals"):
                    creation["evidence_signals"] = [{
                        "title": event.get("title", "Unknown"),
                        "content": event.get("content", "")
                    }]
                current_modules_dict[m_name] = creation
                print(f"  [CREATE] {m_name}")
                
            # Apply merges
            for merge in (response.get("merges") or []):
                ma = safe_get_module(merge.get("module_a"), current_modules_dict)
                mb = safe_get_module(merge.get("module_b"), current_modules_dict)
                if ma and mb and ma != mb:
                    active_modules.add(merge["merged_module"]["name"])
                    active_modules.discard(ma)
                    active_modules.discard(mb)
                    all_evidence = current_modules_dict[ma].get("evidence_signals", []) + current_modules_dict[mb].get("evidence_signals", [])
                    merged_module = merge["merged_module"]
                    
                    all_evidence.append({
                        "title": event.get("title", "Unknown"),
                        "content": event.get("content", "")
                    })
                    merged_module["evidence_signals"] = all_evidence
                    
                    del current_modules_dict[ma]
                    del current_modules_dict[mb]
                    
                    current_modules_dict[merged_module["name"]] = merged_module
                    print(f"  [MERGE] {ma} + {mb} -> {merged_module['name']}")
                    
        except Exception as e:
            print(f"Parser error on event {idx+1}: {e}")
            
    final_modules = list(current_modules_dict.values())
    print(f"\nOUTPUT ({len(final_modules)} synthesized modules): {json.dumps(final_modules, indent=2)}")
    out = {"modules": final_modules, "active_modules": list(active_modules)}
    if state.get("defer_persistence"):
        out["pending_persistence"] = deferred_persistence
    return out
