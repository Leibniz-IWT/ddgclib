# Package manifest

Package name:

`ddg_curved_volume_manuscript_cases_v51`

Base archive:

`ddg_curved_volume_manuscript_cases_v45.7z`

Base Zenodo DOI:

https://doi.org/10.5281/zenodo.19135931

Base archive checksum verified during packaging:

`md5:a5ee62640bd08b701f05352128f517c1`

Main addition relative to v45:

`Figure_3-8_VolumeOperatorComparison/`

The added folder contains:

- scripts to regenerate all Figure 3-8 case CSV files;
- scripts to regenerate Figure 3-8a through Figure 3-8h PNG panels;
- script to regenerate the combined eight-page `Figure_3-8.pdf`;
- script to regenerate the method-labeled mesh-preview PNGs in `mesh/`;
- source data needed by the dynamic cases;
- bundled third-party source trees needed to rebuild the Evrard-type helper,
  with their original license files retained.

Primary reproduction command:

```bash
cd Figure_3-8_VolumeOperatorComparison
python3 recompute_all_cases.py
```

Verified outputs before packaging:

- six static case CSV files with four rows each;
- `cube2sphere_all_methods_result.csv` with 1000 rows;
- `droplet_oscillation_all_methods_result.csv` with 2000 rows;
- `Figure_3-8.pdf` with eight pages;
- 130 method-labeled mesh-preview PNGs.
