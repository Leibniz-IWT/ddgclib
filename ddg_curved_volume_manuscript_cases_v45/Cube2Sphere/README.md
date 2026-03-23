# Cube2Sphere

This folder contains the code and supporting files for the cube-to-sphere relaxation case used in the manuscript archive `ddg_curved_volume_manuscript_cases_v45`.

## Manuscript relevance

This folder corresponds to the cube-to-sphere relaxation benchmark/case reported in the manuscript.

The main script used for the manuscript results in this folder is:

`droplet_to_sphere_v47_Forces_Murnaghan_curved_AiVi_r_ExactR.py`

## Main files

### Main script
- `droplet_to_sphere_v47_Forces_Murnaghan_curved_AiVi_r_ExactR.py`  (with curved volume patch)
  Main script used to generate the manuscript results for the cube-to-sphere relaxation case.

### Direct local dependencies
- `_curvatures_heron.py`  
  Helper module called by the main script.

- `_curved_volume.py`  
  Helper module called by the main script.

### Additional folders
- `curved_volume/`  
  Supporting curved-volume related files.

- `neatplot-main/`  
  Plotting / figure support files used in the workflow.

## Other Python files

The following files are retained for traceability and comparison, but were not the primary script used for the archived manuscript result unless explicitly stated otherwise:

- `droplet_to_sphere_v31_Forces_Murnaghan_flat_r_adjust.py` (without curved volume patch)
- `droplet_to_sphere_v31_Forces_Murnaghan_flat_r_adjust_video_v2.py`

These files represent earlier or alternative variants of the cube-to-sphere workflow.

## Recommended execution

Run the main manuscript script from within this folder so that the local helper modules can be found correctly.

Example:

```bash
python droplet_to_sphere_v47_Forces_Murnaghan_curved_AiVi_r_ExactR.py