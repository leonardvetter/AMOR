"""
STEP → 3D Tetrahedral Mesh Generator (with geometry repair)
=============================================================
Generates coarse and fine 3D tetrahedral meshes from a STEP file.
Handles the common case where STEP files contain shells (no volume)
by attempting multiple repair strategies automatically.

Requirements:
    pip install gmsh meshio numpy

Usage:
    python mesh_3d_generator.py -i aircraft_rib.step --diagnose
    python mesh_3d_generator.py -i aircraft_rib.step --preset neural_operator
    python mesh_3d_generator.py -i aircraft_rib.step --both-meshes
    python mesh_3d_generator.py -i aircraft_rib.step --convergence
"""

import gmsh
import meshio
import numpy as np
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MeshConfig:
    mesh_size_min: float = 0.5
    mesh_size_max: float = 5.0
    algorithm_2d: int = 6       # 6=Frontal-Delaunay (best quality)
    algorithm_3d: int = 1       # 1=Delaunay (robust), 4=Frontal, 10=HXT
    element_order: int = 1      # 1=linear tets (4-node), 2=quadratic (10-node)
    optimize: bool = True
    num_optimize_steps: int = 5
    optimize_threshold: float = 0.3
    min_elements_per_2pi: float = 12.0
    growth_rate: float = 1.5
    geometry_tolerance: float = 1e-4
    healing: bool = True
    tag_boundaries: bool = True


PRESETS = {
    "coarse": MeshConfig(
        mesh_size_min=1.5, mesh_size_max=8.0,
        min_elements_per_2pi=8, growth_rate=1.6, num_optimize_steps=3,
    ),
    "medium": MeshConfig(
        mesh_size_min=1.0, mesh_size_max=6.0,
        min_elements_per_2pi=10, growth_rate=1.5, num_optimize_steps=5,
    ),
    "fine": MeshConfig(
        mesh_size_min=0.8, mesh_size_max=4.0,
        min_elements_per_2pi=12, growth_rate=1.4, num_optimize_steps=5,
    ),
    "neural_operator": MeshConfig(
        mesh_size_min=1.0, mesh_size_max=5.0,
        min_elements_per_2pi=10, growth_rate=1.5, num_optimize_steps=5,
    ),
}


# ============================================================================
# GEOMETRY DIAGNOSTICS
# ============================================================================

def diagnose_step(step_file: str):
    """Full geometry inspection: entities, bounding box, volume check, repair test."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    try:
        print(f"\n{'='*60}")
        print(f"3D GEOMETRY DIAGNOSTICS: {step_file}")
        print(f"{'='*60}")

        _apply_healing_options()
        gmsh.model.occ.importShapes(str(step_file))
        gmsh.model.occ.synchronize()

        points = gmsh.model.getEntities(dim=0)
        curves = gmsh.model.getEntities(dim=1)
        surfaces = gmsh.model.getEntities(dim=2)
        volumes = gmsh.model.getEntities(dim=3)

        print(f"\n  Topological entities:")
        print(f"    Points   (0D): {len(points)}")
        print(f"    Curves   (1D): {len(curves)}")
        print(f"    Surfaces (2D): {len(surfaces)}")
        print(f"    Volumes  (3D): {len(volumes)}")

        # Bounding box
        all_ents = volumes if volumes else surfaces
        if all_ents:
            xmin, ymin, zmin = float('inf'), float('inf'), float('inf')
            xmax, ymax, zmax = float('-inf'), float('-inf'), float('-inf')
            for dim, tag in all_ents:
                bb = gmsh.model.getBoundingBox(dim, tag)
                xmin, ymin, zmin = min(xmin, bb[0]), min(ymin, bb[1]), min(zmin, bb[2])
                xmax, ymax, zmax = max(xmax, bb[3]), max(ymax, bb[4]), max(zmax, bb[5])
            dx, dy, dz = xmax-xmin, ymax-ymin, zmax-zmin
            print(f"\n  Bounding box:")
            print(f"    X: [{xmin:.2f}, {xmax:.2f}]  (size: {dx:.2f})")
            print(f"    Y: [{ymin:.2f}, {ymax:.2f}]  (size: {dy:.2f})")
            print(f"    Z: [{zmin:.2f}, {zmax:.2f}]  (size: {dz:.2f})")

            # Recommended mesh sizes
            char_length = max(dx, dy, dz)
            print(f"\n  Recommended mesh sizes:")
            print(f"    Coarse: min={char_length/50:.2f}, max={char_length/5:.2f}")
            print(f"    Medium: min={char_length/100:.2f}, max={char_length/10:.2f}")
            print(f"    Fine:   min={char_length/200:.2f}, max={char_length/20:.2f}")

        # Diagnosis and repair preview
        print(f"\n  DIAGNOSIS:")
        if len(volumes) > 0:
            print(f"    ✓ Found {len(volumes)} solid volume(s).")
            print(f"    3D tetrahedral meshing should work directly.")
            print(f"    → python mesh_3d_generator.py -i {step_file} --both-meshes")
        elif len(surfaces) > 0:
            print(f"    ✗ Found {len(surfaces)} surfaces but NO volumes.")
            print(f"    Your STEP file has shells, not solids.")
            print(f"\n    Testing automatic repair...")
            success, method = _test_repair(surfaces)
            if success:
                print(f"    ✓ Repair will work (method: {method}).")
                print(f"    → python mesh_3d_generator.py -i {step_file} --both-meshes")
                print(f"      (repair runs automatically)")
            else:
                print(f"    ✗ Automatic repair may fail.")
                print(f"    → Try: --heal-aggressive flag")
                print(f"    → Or use fTetWild on the STL version:")
                print(f"       fTetWild -i aircraft_rib.stl -o rib.msh --epsr 1e-3")
        else:
            print(f"    ✗ No usable geometry found.")

        print(f"{'='*60}\n")

    finally:
        gmsh.finalize()


def _test_repair(surfaces):
    """Test if we can form a volume from the existing surfaces."""
    # Strategy 1: healShapes with makeSolids
    for tol in [1e-4, 1e-3, 1e-2]:
        try:
            gmsh.model.occ.healShapes(
                tolerance=tol, fixDegenerated=True,
                fixSmallEdges=True, fixSmallFaces=True,
                sewFaces=True, makeSolids=True
            )
            gmsh.model.occ.synchronize()
            if len(gmsh.model.getEntities(dim=3)) > 0:
                return True, f"healShapes(tol={tol})"
        except Exception:
            pass

    # Strategy 2: manual surface loop
    try:
        stags = [s[1] for s in gmsh.model.getEntities(dim=2)]
        sl = gmsh.model.occ.addSurfaceLoop(stags)
        gmsh.model.occ.addVolume([sl])
        gmsh.model.occ.synchronize()
        if len(gmsh.model.getEntities(dim=3)) > 0:
            return True, "manual surface loop"
    except Exception:
        pass

    return False, None


# ============================================================================
# GEOMETRY REPAIR
# ============================================================================

def _apply_healing_options():
    """Set OCC healing flags on import."""
    gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
    gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
    gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
    gmsh.option.setNumber("Geometry.OCCSewFaces", 1)


def _ensure_volume(heal_aggressive: bool = False):
    """
    Check for volumes; attempt repair if none found.
    Returns True if at least one volume exists after repair.
    """
    volumes = gmsh.model.getEntities(dim=3)
    if len(volumes) > 0:
        return True

    surfaces = gmsh.model.getEntities(dim=2)
    if len(surfaces) == 0:
        print("  ERROR: No surfaces found.")
        return False

    print(f"  No volumes found. Repairing from {len(surfaces)} surfaces...")

    # Strategy 1: healShapes with makeSolids (increasing tolerance)
    tolerances = [1e-4, 1e-3, 1e-2]
    if heal_aggressive:
        tolerances.extend([5e-2, 1e-1])

    for tol in tolerances:
        try:
            print(f"    healShapes(tol={tol})...", end=" ")
            gmsh.model.occ.healShapes(
                tolerance=tol, fixDegenerated=True,
                fixSmallEdges=True, fixSmallFaces=True,
                sewFaces=True, makeSolids=True
            )
            gmsh.model.occ.synchronize()
            vols = gmsh.model.getEntities(dim=3)
            if len(vols) > 0:
                print(f"✓ Created {len(vols)} volume(s)!")
                return True
            print("no volume yet")
        except Exception as e:
            print(f"failed ({e})")

    # Strategy 2: manual surface loop → volume
    print("    Manual surface loop...", end=" ")
    try:
        stags = [s[1] for s in gmsh.model.getEntities(dim=2)]
        sl = gmsh.model.occ.addSurfaceLoop(stags)
        gmsh.model.occ.addVolume([sl])
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(dim=3)
        if len(vols) > 0:
            print(f"✓ Created {len(vols)} volume(s)!")
            return True
        print("no volume")
    except Exception as e:
        print(f"failed ({e})")

    # Strategy 3: reimport with loose tolerance and retry
    print("    Loose-tolerance reimport...", end=" ")
    try:
        gmsh.option.setNumber("Geometry.Tolerance", 1e-1)
        gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
        gmsh.model.occ.healShapes(
            tolerance=1e-1, fixDegenerated=True,
            fixSmallEdges=True, fixSmallFaces=True,
            sewFaces=True, makeSolids=True
        )
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(dim=3)
        if len(vols) > 0:
            print(f"✓ Created {len(vols)} volume(s)!")
            return True
        print("no volume")
    except Exception as e:
        print(f"failed ({e})")

    print("\n  ✗ All repair strategies failed.")
    print("    Options:")
    print("      1. Fix in FreeCAD: Part → Convert to Solid, then re-export as STEP")
    print("      2. Use the STL version with fTetWild:")
    print("         pip install ftetwild")
    print("         fTetWild -i aircraft_rib.stl -o rib.msh --epsr 1e-3")
    print("      3. Use the 2D script (mesh_generator.py) for plane stress instead")
    return False


# ============================================================================
# 3D MESH GENERATION
# ============================================================================

def generate_3d_mesh(step_file: str, config: MeshConfig,
                     output_prefix: str = None, verbose: bool = True,
                     heal_aggressive: bool = False):
    """Generate a 3D tetrahedral mesh from a STEP file."""
    step_path = Path(step_file)
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_file}")

    if output_prefix is None:
        output_prefix = step_path.stem

    gmsh.initialize()

    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("Geometry.Tolerance", config.geometry_tolerance)

        if config.healing:
            _apply_healing_options()

        if verbose:
            print(f"\nLoading: {step_file}")

        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()

        # ---- Ensure we have a volume ----
        has_volume = _ensure_volume(heal_aggressive)

        if not has_volume:
            print("\n  Cannot generate 3D mesh without a volume.")
            print("  Generating surface mesh only (for inspection in ParaView).")

        volumes = gmsh.model.getEntities(dim=3)
        surfaces = gmsh.model.getEntities(dim=2)

        if verbose:
            print(f"  Entities after repair: {len(surfaces)} surfaces, "
                  f"{len(volumes)} volumes")

        # ---- Mesh parameters ----
        gmsh.option.setNumber("Mesh.MeshSizeMin", config.mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", config.mesh_size_max)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi",
                              config.min_elements_per_2pi)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.Algorithm", config.algorithm_2d)
        gmsh.option.setNumber("Mesh.Algorithm3D", config.algorithm_3d)
        gmsh.option.setNumber("Mesh.ElementOrder", config.element_order)
        gmsh.option.setNumber("Mesh.Optimize", int(config.optimize))
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.OptimizeThreshold", config.optimize_threshold)

        # ---- Physical groups ----
        if config.tag_boundaries and volumes:
            vol_tags = [v[1] for v in volumes]
            gmsh.model.addPhysicalGroup(3, vol_tags, tag=1)
            gmsh.model.setPhysicalName(3, 1, "volume")

            surf_tags = [s[1] for s in surfaces]
            gmsh.model.addPhysicalGroup(2, surf_tags, tag=10)
            gmsh.model.setPhysicalName(2, 10, "boundary")

        # ---- Generate mesh ----
        if verbose:
            print("Generating surface mesh...")
        gmsh.model.mesh.generate(2)

        if has_volume:
            if verbose:
                print("Generating volume mesh...")
            gmsh.model.mesh.generate(3)

            if config.optimize:
                if verbose:
                    print(f"Optimizing ({config.num_optimize_steps} steps)...")
                for _ in range(config.num_optimize_steps):
                    gmsh.model.mesh.optimize("Laplace2D")
                    gmsh.model.mesh.optimize("Relocate3D")
                gmsh.model.mesh.optimize("Netgen")

        # ---- Statistics ----
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        _, elem_tags_3d, _ = gmsh.model.mesh.getElements(dim=3)
        _, elem_tags_2d, _ = gmsh.model.mesh.getElements(dim=2)

        n_nodes = len(node_tags)
        n_tets = sum(len(t) for t in elem_tags_3d)
        n_tris = sum(len(t) for t in elem_tags_2d)

        if n_tets > 0:
            q = np.array(gmsh.model.mesh.getElementQualities(
                list(elem_tags_3d[0]), qualityName="minSJ"))
        else:
            q = np.array([0.0])

        stats = {
            "n_nodes": n_nodes, "n_tets": n_tets, "n_surface_tris": n_tris,
            "quality_min": float(q.min()), "quality_mean": float(q.mean()),
            "quality_median": float(np.median(q)),
            "n_poor_elements": int(np.sum(q < 0.1)),
        }

        # ---- Save .msh ----
        msh_file = f"{output_prefix}.msh"
        gmsh.write(msh_file)
        output_files = {"msh": msh_file}

        config_file = f"{output_prefix}_config.json"
        with open(config_file, "w") as f:
            json.dump(asdict(config), f, indent=2)
        output_files["config"] = config_file

    finally:
        gmsh.finalize()

    # ---- Convert with meshio ----
    mesh = meshio.read(msh_file)

    if n_tets > 0:
        tet_types = ("tetra", "tetra10")
        tet_cells = [c for c in mesh.cells if c.type in tet_types]
        if tet_cells:
            tet_mesh = meshio.Mesh(
                points=mesh.points, cells=tet_cells,
                point_data=mesh.point_data,
                cell_data={k: [v[i] for i, c in enumerate(mesh.cells)
                               if c.type in tet_types]
                           for k, v in mesh.cell_data.items()}
                          if mesh.cell_data else {}
            )
            for ext in (".vtu", ".xdmf"):
                fname = f"{output_prefix}{ext}"
                meshio.write(fname, tet_mesh)
                output_files[ext.strip(".")] = fname

            # Numpy arrays for neural operator
            np.savez(f"{output_prefix}_arrays.npz",
                     nodes=mesh.points,
                     elements=tet_cells[0].data,
                     n_nodes=len(mesh.points),
                     n_elements=len(tet_cells[0].data))
            output_files["npz"] = f"{output_prefix}_arrays.npz"
    else:
        # Surface mesh only
        tri_cells = [c for c in mesh.cells if c.type == "triangle"]
        if tri_cells:
            tri_mesh = meshio.Mesh(points=mesh.points, cells=tri_cells)
            fname = f"{output_prefix}_surface.vtu"
            meshio.write(fname, tri_mesh)
            output_files["surface_vtu"] = fname

    # ---- Summary ----
    if verbose:
        print(f"\n{'='*60}")
        print("3D MESH STATISTICS")
        print(f"{'='*60}")
        print(f"  Nodes:              {stats['n_nodes']:,}")
        print(f"  Tetrahedra:         {stats['n_tets']:,}")
        print(f"  Surface triangles:  {stats['n_surface_tris']:,}")
        if n_tets > 0:
            print(f"  Quality (min):      {stats['quality_min']:.4f}")
            print(f"  Quality (mean):     {stats['quality_mean']:.4f}")
            print(f"  Quality (median):   {stats['quality_median']:.4f}")
            print(f"  Poor elements (<0.1): {stats['n_poor_elements']:,} "
                  f"({100*stats['n_poor_elements']/max(n_tets,1):.1f}%)")
        else:
            print(f"  ⚠  NO TETRAHEDRA — volume repair failed")
            print(f"     See suggestions above or run --diagnose")
        print(f"  Element order:      {config.element_order}")
        print(f"  Files: {list(output_files.keys())}")
        print(f"{'='*60}")

    stats["output_files"] = output_files
    return stats


# ============================================================================
# COARSE + FINE PAIR
# ============================================================================

def generate_mesh_pair(step_file: str, output_dir: str = "meshes_3d",
                       coarse_preset: str = "coarse",
                       fine_preset: str = "medium",
                       heal_aggressive: bool = False):
    """Generate coarse + fine 3D mesh pair for multi-fidelity training."""
    Path(output_dir).mkdir(exist_ok=True)

    print("=" * 60)
    print("GENERATING 3D COARSE + FINE MESH PAIR")
    print(f"  Coarse preset: {coarse_preset}")
    print(f"  Fine preset:   {fine_preset}")
    print("=" * 60)

    print(f"\n--- COARSE MESH ({coarse_preset}) ---")
    coarse = generate_3d_mesh(step_file, PRESETS[coarse_preset],
                              output_prefix=f"{output_dir}/rib_coarse",
                              heal_aggressive=heal_aggressive)

    print(f"\n--- FINE MESH ({fine_preset}) ---")
    fine = generate_3d_mesh(step_file, PRESETS[fine_preset],
                            output_prefix=f"{output_dir}/rib_fine",
                            heal_aggressive=heal_aggressive)

    print(f"\n{'='*60}")
    print("3D MESH PAIR SUMMARY")
    print(f"{'='*60}")
    print(f"  {'':20}{'Coarse':>12}{'Fine':>12}{'Ratio':>10}")
    print(f"  {'-'*54}")
    nc, nf = coarse['n_nodes'], fine['n_nodes']
    tc, tf = coarse['n_tets'], fine['n_tets']
    print(f"  {'Nodes':<20}{nc:>12,}{nf:>12,}{nf/max(nc,1):>10.1f}x")
    print(f"  {'Tetrahedra':<20}{tc:>12,}{tf:>12,}{tf/max(tc,1):>10.1f}x")
    print(f"  {'Quality (mean)':<20}{coarse['quality_mean']:>12.4f}"
          f"{fine['quality_mean']:>12.4f}")
    print(f"{'='*60}")
    print(f"\n  Coarse FEM DOFs (3D displacement): ~{3*nc:,}")
    print(f"  Fine FEM DOFs (3D displacement):   ~{3*nf:,}")
    print(f"\n  Estimated FEM solve time per sample (hyperelasticity, Newton):")
    print(f"    Coarse: ~{max(0.5, 3*nc/30000):.1f}s  "
          f"→ 5k samples ≈ {max(1, 3*nc/30000 * 5000 / 3600):.0f}h")
    print(f"    Fine:   ~{max(1, 3*nf/30000):.1f}s  "
          f"→ 5k samples ≈ {max(1, 3*nf/30000 * 5000 / 3600):.0f}h")
    print(f"\n  TIP: If fine mesh solve times are too long, consider:")
    print(f"    - Generating training data on coarse, fine-tune on fine")
    print(f"    - Using multi-fidelity neural operators")
    print(f"    - Parallel FEM solves (MPI with FEniCSx)")

    return coarse, fine


# ============================================================================
# CONVERGENCE STUDY
# ============================================================================

def convergence_study(step_file: str, output_dir: str = "convergence_3d",
                      heal_aggressive: bool = False):
    """3D mesh convergence study."""
    levels = [
        ("coarse",      1.5, 8.0,  8),
        ("medium",      1.0, 6.0,  10),
        ("fine",        0.8, 4.0,  12),
    ]

    Path(output_dir).mkdir(exist_ok=True)
    results = []

    for name, sz_min, sz_max, el2pi in levels:
        print(f"\n{'─'*60}")
        print(f"Level: {name} (h=[{sz_min}, {sz_max}])")
        print(f"{'─'*60}")
        cfg = MeshConfig(mesh_size_min=sz_min, mesh_size_max=sz_max,
                         min_elements_per_2pi=el2pi)
        stats = generate_3d_mesh(step_file, cfg,
                                 output_prefix=f"{output_dir}/mesh_{name}",
                                 heal_aggressive=heal_aggressive)
        stats["name"] = name
        stats["mesh_size_min"] = sz_min
        stats["mesh_size_max"] = sz_max
        results.append(stats)

    print(f"\n\n{'='*72}")
    print("3D MESH CONVERGENCE SUMMARY")
    print(f"{'─'*72}")
    print(f"{'Level':<14}{'min_h':<8}{'max_h':<8}{'Nodes':<12}"
          f"{'Tets':<12}{'Q_mean':<10}")
    print(f"{'─'*72}")
    for r in results:
        print(f"{r['name']:<14}{r['mesh_size_min']:<8.1f}"
              f"{r['mesh_size_max']:<8.1f}{r['n_nodes']:<12,}"
              f"{r['n_tets']:<12,}{r['quality_mean']:<10.4f}")
    print(f"{'='*72}")

    if all(r['n_tets'] == 0 for r in results):
        print(f"\n  ⚠  All levels: 0 tetrahedra. Volume repair failed.")
        print(f"  Run with --heal-aggressive or try fTetWild.")

    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="STEP → 3D Tetrahedral Mesh Generator")
    p.add_argument("--input", "-i", type=str, default="aircraft_rib.step")
    p.add_argument("--output", "-o", type=str, default=None)
    p.add_argument("--preset", "-p", type=str, default="neural_operator",
                   choices=list(PRESETS.keys()))
    p.add_argument("--min-size", type=float, default=None)
    p.add_argument("--max-size", type=float, default=None)
    p.add_argument("--order", type=int, default=None)
    p.add_argument("--heal-aggressive", action="store_true",
                   help="Try more aggressive geometry repair tolerances")
    p.add_argument("--coarse-preset", type=str, default="coarse",
                   choices=list(PRESETS.keys()),
                   help="Preset for coarse mesh (default: coarse)")
    p.add_argument("--fine-preset", type=str, default="medium",
                   choices=list(PRESETS.keys()),
                   help="Preset for fine mesh (default: medium)")

    p.add_argument("--diagnose", action="store_true",
                   help="Inspect geometry and test if repair works")
    p.add_argument("--both-meshes", action="store_true",
                   help="Generate coarse + fine mesh pair")
    p.add_argument("--convergence", action="store_true",
                   help="Run mesh convergence study")

    args = p.parse_args()

    if args.diagnose:
        diagnose_step(args.input)
    elif args.both_meshes:
        generate_mesh_pair(args.input,
                           coarse_preset=args.coarse_preset,
                           fine_preset=args.fine_preset,
                           heal_aggressive=args.heal_aggressive)
    elif args.convergence:
        convergence_study(args.input, heal_aggressive=args.heal_aggressive)
    else:
        config = PRESETS[args.preset]
        if args.min_size is not None:
            config.mesh_size_min = args.min_size
        if args.max_size is not None:
            config.mesh_size_max = args.max_size
        if args.order is not None:
            config.element_order = args.order

        generate_3d_mesh(args.input, config, output_prefix=args.output,
                         heal_aggressive=args.heal_aggressive)
