# ddg_curved_volume_manuscript_cases_v51

This archive extends the released Zenodo v45 code snapshot with the additional
volume-operator comparison used in manuscript version `v51`.

The original v45 package is:

Deng, S., Endres, S. C., and Maedler, L. (2026). *DDG curved-volume manuscript
cases*, version v45. Zenodo. https://doi.org/10.5281/zenodo.19135931

## Contents

This archive is organized into four case folders:

- `StaticValidationGeometries/`  
  Static validation cases on canonical geometries.

- `Cube2Sphere/`  
  Cube-to-sphere relaxation case.

- `OscillatingDroplet/`  
  Oscillating droplet benchmark case.

- `Figure_3-8_VolumeOperatorComparison/`  
  Reproducible code, data, generated CSVs, PNGs, and PDF for the Figure 3-8
  volume-operator comparison added during rebuttal revision.

## Purpose of this archive

This archive preserves the code snapshot used for the manuscript results associated with manuscript version `v51`.

The folders correspond to the main computational case groups reported in the manuscript:
1. Static validation geometries
2. Cube-to-sphere relaxation
3. Oscillating droplet benchmark
4. Figure 3-8 volume-operator comparison

## Folder overview

### `StaticValidationGeometries/`
Contains the notebook, helper module, mesh files, and plotting/support files used for the static validation cases.

Main file:
- `manuscript_curved_volume_final.ipynb`

### `Cube2Sphere/`
Contains the main script, helper modules, and support files used for the cube-to-sphere relaxation case.

Main file:
- `droplet_to_sphere_v47_Forces_Murnaghan_curved_AiVi_r_ExactR.py`

### `OscillatingDroplet/`
Contains the main script, helper modules, and support files used for the oscillating droplet case.

Main file:
- `_AR1.05_dyn_mi_finer0.15_NoGuard_NoCap_GMSHsnap_cotan_v11_Pgrad 0.33_1.5mass_0Ai.py`

### `Figure_3-8_VolumeOperatorComparison/`
Contains the scripts and source data used to regenerate the Figure 3-8
comparison CSVs, eight PNG panels, combined PDF, and mesh-preview PNGs.

Main file:
- `recompute_all_cases.py`

Plot-only file:
- `Figure_3-8_VolumeOperatorComparison.py`

Mesh-preview file:
- `mesh_plot.py`

## Notes on file organization

Each case folder may also contain:
- local helper modules
- plotting utilities
- mesh or geometry files
- earlier or alternative development variants kept for traceability

Please see the `README.md` inside each case folder for case-specific details.

## Third-party code and licenses

Most author-created scripts are distributed under the archive license used for
the released manuscript-code package. Some source trees bundled inside
`Figure_3-8_VolumeOperatorComparison/` are third-party code used only for
reproducibility of comparison operators. These third-party source trees retain
their original licenses and are documented in `THIRD_PARTY_NOTICE.md`.

The Python dependency folder `_python_deps/` is intentionally not included.
Dependencies are installed from `requirements.txt` into a local folder when the
recompute scripts are run.

## Reproducibility note

This archive is intended to preserve the exact manuscript code snapshot for the reported cases.  
For publication citation, please cite the archived Zenodo record corresponding to this archive.

## Suggested citation

Deng, S., Endres, S. C., and Maedler, L. (2026). *DDG curved-volume manuscript
cases* (Version v51) [Computer software]. Zenodo.
https://doi.org/10.5281/zenodo.21181878
