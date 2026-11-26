import os
import json

def compact_module_label(name: str, provisional: bool = False, max_chars: int = 34, max_lines: int = 3) -> str:
    words = name.split()
    lines = []
    current = "(P) " if provisional else ""
    used_count = 0

    for index, word in enumerate(words):
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
        used_count = index + 1
        if len(lines) >= max_lines:
            break

    if current:
        lines.append(current)

    if used_count < len(words):
        lines[-1] = lines[-1].rstrip(" .,;:-") + "..."
    return "\n".join(lines[:max_lines])


def visualize_graph_pyvis(horizon_map: dict, modules: list = None):
    try:
        from pyvis.network import Network
    except ImportError:
        print("\n  [Visualization] 'pyvis' not installed!!!")
        return
        
    net = Network(height="720px", width="100%", bgcolor="#ffffff", font_color="#1f2937", directed=True)
    
    modules = modules or []
    module_details = {}
    for m in modules:
        name = m.get("name", "Unknown")
        reasoning = m.get("reasoning", "")
        evidence = m.get("evidence_signals", [])
        
        tooltip = ("(P) " + name + "\nEmerging module: provisional, supported by limited evidence\n"
                   if m.get("provisional") else name + "\n")
        tooltip += "\n"

        tooltip += f"Reasoning:\n{reasoning}\n\n"
        
        if evidence:
            tooltip += "Signals supporting this module:\n"
            for ev in evidence:
                title = ev.get('title', '')
                if title:
                    tooltip += f"  • {title}\n"
            
        module_details[name] = tooltip
    
    provisional = {m.get("name") for m in modules if m.get("provisional")}
    module_colors = {}

    # First, calculate in-degree and out-degree for all nodes
    degrees = {}
    for edge in horizon_map.get("knowledge_graph_edges", []):
        src = edge.get("source_module", "Unknown")
        tgt = edge.get("target_module", "Unknown")
        if src not in degrees:
            degrees[src] = {"in": 0, "out": 0}
        if tgt not in degrees:
            degrees[tgt] = {"in": 0, "out": 0}
        degrees[src]["out"] += 1
        degrees[tgt]["in"] += 1

    # Ensure all listed modules are accounted for
    for m in modules:
        name = m.get("name", "Unknown")
        if name not in degrees:
            degrees[name] = {"in": 0, "out": 0}

    # Add Nodes based on their systemic degree role
    for name, degs in degrees.items():
        in_d = degs["in"]
        out_d = degs["out"]
        
        # Style based on network properties
        if in_d > 0 and out_d > 0:
            # Feedback / Transitive Hub
            color = "#7b2cbf"  # Deep Royal Purple / Amethyst
            size = 30          # Larger to emphasize systemic brokerage
        elif out_d > 0:
            # Pure Source / Risk Driver
            color = "#00d8ff"  # Bright Cyan
            size = 24
        elif in_d > 0:
            # Pure Target / Vulnerability
            color = "#bd00ff"  # Vibrant Magenta
            size = 24
        else:
            # Isolated Module
            color = "#888888"  # Slate Grey
            size = 20

        if name in provisional:
            # Amber dashed borders mark provisional nodes.
            net.add_node(name, label=compact_module_label(name, provisional=True), title="", size=size, borderWidth=3,
                         color={"background": "#fff7d6", "border": "#b7791f"},
                         shapeProperties={"borderDashes": [6, 4]})
        else:
            net.add_node(name, label=compact_module_label(name), title="", color=color, size=size)
        module_colors[name] = "#b7791f" if name in provisional else color

    # 2. Add all the edges sequentially after all nodes are created
    edge_details = {}
    for edge_index, edge in enumerate(horizon_map.get("knowledge_graph_edges", [])):
        src = edge.get("source_module", "Unknown")
        tgt = edge.get("target_module", "Unknown")
        
        # Guard: Ensure source and target actually exist in PyVis nodes
        if src not in net.get_nodes():
            net.add_node(src, label=compact_module_label(src), color="#888888", size=20)
            module_colors[src] = "#888888"
        if tgt not in net.get_nodes():
            net.add_node(tgt, label=compact_module_label(tgt), color="#888888", size=20)
            module_colors[tgt] = "#888888"
            
        edge_label = edge.get("interaction_type", "INTERACTS")
        weight = edge.get("empirical_trend_strength", 1)
        narrative = edge.get("narrative_salience", "-")
        inst = edge.get("institutional_relevance", "-")
        causality = edge.get("causal_directionality", "Unidirectional")
        
        # Determine causality arrows and symbol
        arrows = "to"
        arrow_char = "➡️"
        if causality.lower().startswith("bi"):
            arrows = "to, from"
            arrow_char = "🔁"
            
        edge_tooltip = f"{src} {arrow_char} {tgt}\n"
        edge_tooltip += f"Relationship: {edge_label}\n"
        edge_tooltip += f"Direction: {causality}\n"
        edge_tooltip += f"Empirical trend strength: {weight}/10\n"
        edge_tooltip += f"Narrative salience: {narrative}/10\n"
        edge_tooltip += f"Institutional relevance: {inst}/10"

        edge_id = f"{src}__{tgt}__{edge_index}"
        edge_details[edge_id] = edge_tooltip
        net.add_edge(src, tgt, id=edge_id, title="", label=str(weight), value=weight, arrows=arrows,
                     color="#c58a18" if edge.get("provisional") else "#6b7280", dashes=bool(edge.get("provisional")))
        
    # 2. Physics options nice layout
    net.set_options("""
    var options = {
      "nodes": {
        "font": { "size": 14, "color": "#1f2937", "multi": false, "align": "middle" }
      },
      "edges": {
        "font": { "color": "#374151", "size": 12, "align": "middle" }
      },
      "interaction": {
        "tooltipDelay": 100,
        "hover": true,
        "selectable": true,
        "navigationButtons": true,
        "keyboard": true
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -120,
          "centralGravity": 0.01,
          "springLength": 250,
          "springConstant": 0.05
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35
      }
    }
    """)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "out")
    os.makedirs(out_dir, exist_ok=True)
    graph_path = os.path.join(out_dir, "horizon_graph.html")
    
    try:
        net.write_html(graph_path)

        module_details_json = json.dumps(module_details, ensure_ascii=False)
        module_colors_json = json.dumps(module_colors, ensure_ascii=False)
        edge_details_json = json.dumps(edge_details, ensure_ascii=False)

        enhancement = f"""
<style>
  #mynetwork {{ height: 720px !important; border: 1px solid #d9e2ec; border-radius: 8px 8px 0 0; }}
  #graph-hover-details {{
    display: block;
    min-height: 96px;
    height: auto;
    box-sizing: border-box;
    width: 100%;
    max-width: none;
    margin: 0 0 8px 0;
    padding: 12px 14px;
    border: 1px solid #d9e2ec;
    border-top: 0;
    border-radius: 0 0 8px 8px;
    background: #f8fafc;
    color: #1f2937;
    font-family: Arial, sans-serif;
    font-size: 14px;
    line-height: 1.45;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    word-break: normal;
  }}
  div.vis-tooltip {{ display: none !important; }}
  #graph-select-instruction {{
    width: 100%;
    box-sizing: border-box;
    text-align: center;
    margin: 8px 0;
    color: #4b5563;
    font-family: Arial, sans-serif;
    font-size: 14px;
  }}
</style>
<div id="graph-select-instruction">Select a module or a connection to get more information about it.</div>
<div id="graph-hover-details">No module or connection selected.</div>
<script>
  const graphModuleDetails = {module_details_json};
  const graphModuleColors = {module_colors_json};
  const graphEdgeDetails = {edge_details_json};
  const graphDetails = document.getElementById("graph-hover-details");
  let detailsPinned = false;

  function resizeDetailFrame() {{
    if (!window.frameElement) return;
    requestAnimationFrame(function () {{
      const graphHeight = document.getElementById("mynetwork").getBoundingClientRect().height;
      const detailHeight = graphDetails.getBoundingClientRect().height;
      const legendSpacing = 16;
      window.frameElement.style.height =
        Math.ceil(graphHeight + detailHeight + legendSpacing) + "px";
    }});
  }}

  const detailResizeObserver = new ResizeObserver(resizeDetailFrame);
  detailResizeObserver.observe(graphDetails);

  function clearGraphDetails() {{
    detailsPinned = false;
    graphDetails.textContent = "No module or connection selected.";
    graphDetails.style.border = "1px solid #d9e2ec";
    graphDetails.style.borderTop = "0";
    resizeDetailFrame();
  }}

  function showModuleDetails(moduleName) {{
    graphDetails.textContent = graphModuleDetails[moduleName] || moduleName;
    graphDetails.style.border = "2px solid " + (graphModuleColors[moduleName] || "#888888");
    graphDetails.style.borderTop = "0";
    resizeDetailFrame();
  }}

  function showEdgeDetails(edgeId) {{
    graphDetails.textContent = graphEdgeDetails[edgeId] || "Connection details unavailable.";
    graphDetails.style.border = "2px solid #6b7280";
    graphDetails.style.borderTop = "0";
    resizeDetailFrame();
  }}

  if (typeof network !== "undefined") {{
    network.on("click", function (params) {{
      if (params.nodes && params.nodes.length) {{
        detailsPinned = true;
        showModuleDetails(params.nodes[0]);
      }} else if (params.edges && params.edges.length) {{
        detailsPinned = true;
        showEdgeDetails(params.edges[0]);
      }} else {{
        detailsPinned = false;
        clearGraphDetails();
      }}
    }});
  }}
</script>
"""

        with open(graph_path, "r", encoding="utf-8") as graph_file:
            graph_html = graph_file.read()
        graph_html = graph_html.replace("</body>", enhancement + "</body>", 1)
        with open(graph_path, "w", encoding="utf-8") as graph_file:
            graph_file.write(graph_html)

        print(f"\n  ✨ [Visualization] Interactive Graph saved to '{graph_path}' (Open it in your browser!) ✨")
    except Exception as e:
        print(f"Error saving graph: {e}")
