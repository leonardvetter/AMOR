# Aircraft rib data pipeline

Generators for the aircraft-rib linear-elasticity NPZ datasets.
**Dataset binaries are not in git**; download from the project Zenodo release.

The rib STEP CAD file is **not** redistributed in this repository (third-party geometry cited in the paper). Place a local copy if you rebuild the mesh, e.g. as `$AMOR_MESHES_DIR/aircraft_rib.step`.

Env: `AMOR_DATA_ROOT`, `AMOR_SCRATCH_DATA`, `AMOR_MESHES_DIR`, `AMOR_RIB_DATASETS` (`AMOR/paths.py`).

## Outputs (Zenodo / local data root)

| File | Role |
|---|---|
| `aircraft_rib_cantilever_tip.npz` | Tip Neumann, scalar load |
| `aircraft_rib_cantilever_tip_loadfield.npz` | Tip in bcfield layout |
| `aircraft_rib_distributed_aero.npz` | Distributed traction |
| `aircraft_rib_multipatch_bracket.npz` | Localized patches |
| `aircraft_rib_mixed_loading_combined.npz` | Pooled mixed dataset |

## Rebuild mesh / samples (optional)

```bash
cd AMOR/data/aircraft_rib
python3 mesh_3D_generator.py -i /path/to/aircraft_rib.step --preset neural_operator \
  -o $AMOR_MESHES_DIR/rib_h15.0
python3 boundary_marking.py --npz $AMOR_MESHES_DIR/rib_h15.0_arrays.npz \
  --export --output-dir $AMOR_MESHES_DIR
# then generate_* scripts; see --help on each
```

Training entry points: root `README.md`.
