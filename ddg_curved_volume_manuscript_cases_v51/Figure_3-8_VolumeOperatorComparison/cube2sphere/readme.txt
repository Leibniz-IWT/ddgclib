cube2sphere

This folder regenerates the dynamic Figure 3-8 panel from the original cube-to-sphere simulation artifacts, not from a plotting-only CSV. The PL and present quadric-patch curves come from the simulation summaries; the Evrard-type, THINC/QQ-type, and Strobl-type curves are recomputed from the saved surface mesh states.

Run all methods from this folder:

python cube2sphere_all_methods.py

This writes cube2sphere_all_methods_result.csv and method-specific result files under the method folders. Full recomputation may be slow because it evaluates every saved mesh state.
