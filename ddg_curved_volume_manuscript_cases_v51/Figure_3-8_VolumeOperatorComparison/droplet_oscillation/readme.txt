droplet_oscillation

This folder regenerates the dynamic Figure 3-8 panel from the original oscillating-droplet simulation artifacts, not from a plotting-only CSV. The PL and present quadric-patch curves come from the simulation summaries; the Evrard-type, THINC/QQ-type, and Strobl-type curves are recomputed from the saved surface mesh states.

Run all methods from this folder:

python droplet_oscillation_all_methods.py

This writes droplet_oscillation_all_methods_result.csv and method-specific result files under the method folders. Full recomputation may be slow because it evaluates every saved mesh state.
