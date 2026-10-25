from cpnpy.visualization.visualizer import *  # for format_token, summarize_label, etc.
from cpnpy.cpn.cpn_imp import *              # for CPN, Place, Transition, Arc, Marking
from cpnpy.hcpn.hcpn_imp import *            # for HCPN

import graphviz
import tempfile
import os

# ------------------------------------------------------------------------
# HCPN Visualization
# ------------------------------------------------------------------------
class HCPNGraphViz:
    """
    Visualization for a Hierarchical Coloured Petri Net (HCPN).
    - Each module is displayed as a subgraph (cluster).
    - Substitution transitions are highlighted with a special color and
      a label indicating their submodule.
    - Places and transitions are named with "ModuleName_PlaceName" or
      "ModuleName_TransitionName" for uniqueness in the overall diagram.
    - We draw dashed arcs **between** a parent's substitution transition
      and all transitions in the child's module, visually chaining
      the hierarchy (instead of linking to the entire cluster).
    """

    def __init__(self):
        self.graph = None
        self.format = "pdf"
        self.temp_dir = tempfile.mkdtemp()

        # We'll store the node IDs we assign to each place/transition.
        # Key: (module_name, place_name) or (module_name, transition_name)
        # Value: the unique node_id in Graphviz
        self.node_id_place = {}
        self.node_id_trans = {}

    def apply(
        self,
        hcpn: HCPN,
        markings: Dict[str, Marking],
        format: str = "pdf"
    ):
        """
        Create a Graphviz Digraph from the given HCPN and a dictionary of markings
        (one for each module). If a module is missing from `markings`, it is displayed
        with no tokens.
        """
        self.format = format
        self.graph = graphviz.Digraph(format=self.format, directory=self.temp_dir)
        self.graph.attr(rankdir="LR")

        # 1) Draw each module as a subgraph
        for module_name, cpn in hcpn.modules.items():

            # Create a subgraph (cluster) for each module
            sub = graphviz.Digraph(name=f"cluster_{module_name}")
            sub.attr(label=module_name)
            sub.attr(style="dashed")  # helps visualize module boundaries

            # Get the marking for this module (or empty if missing)
            marking = markings.get(module_name, Marking())

            # --- Places ---
            for place in cpn.places:
                node_id = f"{module_name}__P__{place.name}"
                self.node_id_place[(module_name, place.name)] = node_id

                # Gather tokens from marking
                ms = marking.get_multiset(place.name)
                timed = bool(getattr(place.colorset, "timed", False))
                token_str_list = [
                    format_token(tok, include_timestamp=timed) for tok in ms.tokens
                ]
                if token_str_list:
                    label = f"{place.name}\\nTokens: {', '.join(token_str_list)}"
                else:
                    label = f"{place.name}\\n(No tokens)"
                label = summarize_label(label)

                sub.node(
                    node_id,
                    label=label,
                    shape="ellipse",
                    style="filled",
                    fillcolor="#e0e0f0"
                )

            # --- Transitions ---
            for transition in cpn.transitions:
                node_id = f"{module_name}__T__{transition.name}"
                self.node_id_trans[(module_name, transition.name)] = node_id

                lines = [transition.name]

                # Guard
                if transition.guard_expr:
                    guard_escaped = summarize_label(transition.guard_expr, max_len=500)
                    lines.append(f"Guard: {guard_escaped}")

                # Variables
                if transition.variables:
                    vars_str = ", ".join(
                        summarize_label(v, max_len=100) for v in transition.variables
                    )
                    lines.append(f"Vars: {vars_str}")

                # Delay
                if transition.transition_delay > 0:
                    lines.append(f"Delay: {transition.transition_delay}")

                # Check if this transition is a substitution transition
                target_submodule = hcpn.get_substitution_target(module_name, transition.name)
                if target_submodule is not None:
                    lines.append(f"[Substitutes -> {target_submodule}]")
                    fillcolor = "#ffc080"  # highlight substitution transitions
                else:
                    fillcolor = "#ffe0e0"

                raw_label = "\\n".join(lines)
                final_label = summarize_label(raw_label)

                sub.node(
                    node_id,
                    label=final_label,
                    shape="rectangle",
                    style="rounded,filled",
                    fillcolor=fillcolor
                )

            # --- Arcs (within the module) ---
            for arc in cpn.arcs:
                source_id = None
                target_id = None

                if isinstance(arc.source, Place):
                    source_id = self.node_id_place[(module_name, arc.source.name)]
                elif isinstance(arc.source, Transition):
                    source_id = self.node_id_trans[(module_name, arc.source.name)]

                if isinstance(arc.target, Place):
                    target_id = self.node_id_place[(module_name, arc.target.name)]
                elif isinstance(arc.target, Transition):
                    target_id = self.node_id_trans[(module_name, arc.target.name)]

                arc_label = summarize_label(str(arc.expression), max_len=500)

                if source_id and target_id:
                    sub.edge(source_id, target_id, label=arc_label)

            self.graph.subgraph(sub)

        # 2) Draw dashed arcs from parent's substitution transition -> child's transitions
        for (parent_mod, parent_trans), child_mod in hcpn.substitutions.items():

            # Parent sub-transition node (in parent's module)
            parent_node_id = self.node_id_trans.get((parent_mod, parent_trans))
            if not parent_node_id:
                continue  # can't draw an edge if the node wasn't found

            # Instead of linking only to the child's *substitution* transitions,
            # link to ALL transitions in the child's module:
            child_cpn = hcpn.modules[child_mod]
            for t in child_cpn.transitions:
                child_node_id = self.node_id_trans.get((child_mod, t.name))
                if child_node_id:
                    self.graph.edge(
                        parent_node_id,
                        child_node_id,
                        style="dashed",
                        label="(sub->child)"
                    )

        return self

    def view(self):
        """
        View the generated graph with the system's default viewer.
        """
        if self.graph is None:
            raise RuntimeError("Graph not created. Call apply() first.")
        self.graph.view()

    def save(self, filename: str):
        """
        Save (render) the graph to a file (e.g. 'my_hcpn_graph').
        The format is determined by self.format (pdf, png, etc.).
        """
        if self.graph is None:
            raise RuntimeError("Graph not created. Call apply() first.")
        out_path = self.graph.render(filename=filename, cleanup=True)
        final_path = os.path.join(os.getcwd(), os.path.basename(out_path))
        if os.path.abspath(final_path) != os.path.abspath(out_path):
            os.rename(out_path, final_path)
        return final_path


# ------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------
if __name__ == "__main__":

    # As before, define some color sets, places, transitions, arcs, modules, etc.
    cs_definitions = """
    colset INT = int timed;
    """
    parser = ColorSetParser()
    colorsets = parser.parse_definitions(cs_definitions)
    int_set = colorsets["INT"]

    # MODULE D
    cpn_D = CPN()
    pD_in = Place("P_D_in", int_set)
    pD_out = Place("P_D_out", int_set)
    tD = Transition("T_D", variables=["d"], guard="d >= 0", transition_delay=1)
    cpn_D.add_place(pD_in)
    cpn_D.add_place(pD_out)
    cpn_D.add_transition(tD)
    cpn_D.add_arc(Arc(pD_in, tD, "d"))
    cpn_D.add_arc(Arc(tD, pD_out, "d+5"))

    # MODULE C
    cpn_C = CPN()
    pC_in = Place("P_C_in", int_set)
    pC_mid = Place("P_C_mid", int_set)
    pC_out = Place("P_C_out", int_set)
    tC = Transition("T_C", variables=["c"], guard="c < 100", transition_delay=0)
    tCSub = Transition("T_CSub", variables=["c2"])  # references D
    cpn_C.add_place(pC_in)
    cpn_C.add_place(pC_mid)
    cpn_C.add_place(pC_out)
    cpn_C.add_transition(tC)
    cpn_C.add_transition(tCSub)
    cpn_C.add_arc(Arc(pC_in, tC, "c"))
    cpn_C.add_arc(Arc(tC, pC_mid, "c+10"))
    cpn_C.add_arc(Arc(pC_mid, tCSub, "c2"))
    cpn_C.add_arc(Arc(tCSub, pC_out, "c2*2"))

    # MODULE B
    cpn_B = CPN()
    pB_in = Place("P_B_in", int_set)
    pB_pass = Place("P_B_pass", int_set)
    tB = Transition("T_B", variables=["b"], guard="b >= 0", transition_delay=2)
    tBSub = Transition("T_BSub", variables=["b2"])  # references C
    cpn_B.add_place(pB_in)
    cpn_B.add_place(pB_pass)
    cpn_B.add_transition(tB)
    cpn_B.add_transition(tBSub)
    cpn_B.add_arc(Arc(pB_in, tB, "b"))
    cpn_B.add_arc(Arc(tB, pB_pass, "b+1"))
    cpn_B.add_arc(Arc(pB_pass, tBSub, "b2"))
    # after going through C, tokens re-enter B_in
    cpn_B.add_arc(Arc(tBSub, pB_in, "b2-5"))

    # MODULE A
    cpn_A = CPN()
    pA_start = Place("P_A_start", int_set)
    pA_mid = Place("P_A_mid", int_set)
    pA_fused = Place("P_A_fused", int_set)
    tA = Transition("T_A", variables=["a"], guard="a >= 0", transition_delay=0)
    tASub = Transition("T_ASub", variables=["a2"])  # references B
    cpn_A.add_place(pA_start)
    cpn_A.add_place(pA_mid)
    cpn_A.add_place(pA_fused)
    cpn_A.add_transition(tA)
    cpn_A.add_transition(tASub)
    cpn_A.add_arc(Arc(pA_start, tA, "a"))
    cpn_A.add_arc(Arc(tA, pA_fused, "a*2"))
    cpn_A.add_arc(Arc(pA_fused, tASub, "a2"))
    cpn_A.add_arc(Arc(tASub, pA_mid, "a2+3"))

    # Build the HCPN
    hcpn = HCPN()
    hcpn.add_module("A", cpn_A)
    hcpn.add_module("B", cpn_B)
    hcpn.add_module("C", cpn_C)
    hcpn.add_module("D", cpn_D)

    # Substitutions
    hcpn.add_substitution("A", "T_ASub", "B")
    hcpn.add_substitution("B", "T_BSub", "C")
    hcpn.add_substitution("C", "T_CSub", "D")

    # Initial Markings
    marking_A = Marking()
    marking_A.set_tokens("P_A_start", [0, 10, 20])

    marking_B = Marking()
    marking_C = Marking()
    marking_D = Marking()

    markings_dict = {
        "A": marking_A,
        "B": marking_B,
        "C": marking_C,
        "D": marking_D
    }

    # Visualize the entire HCPN: parent's sub transition -> child's transitions
    viz = HCPNGraphViz().apply(hcpn, markings_dict, format="pdf")

    # View the generated PDF (uncomment if you want to open a viewer)
    viz.view()

    # Or save to a file (uncomment to save)
    # path = viz.save("hcpn_substitution_chain")
    # print("Saved to:", path)

    print("HCPN structure:")
    print(hcpn)
    print("\nMarkings:")
    for mod_name, mark in markings_dict.items():
        print(f"Module {mod_name}:\n{mark}")
