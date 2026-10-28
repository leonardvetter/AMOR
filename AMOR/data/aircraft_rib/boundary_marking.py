"""
Aircraft Rib: Boundary Marking & Linear Elasticity Problem Setup
================================================================
Identifies Γ_D (clamped spar face), Γ_N (loaded trailing edge face),
and Γ_free (everything else = do-nothing) on a 3D tet mesh.

Exports a marked mesh ready for FEM forward solves.

Physical setup:
    Γ_D:    u = 0                (clamped at spar attachment, blunt end)
    Γ_N:    σ·n = t_k            (applied traction at trailing edge tip)
    Γ_free: σ·n = 0              (traction-free everywhere else)

Requirements:
    pip install gmsh meshio numpy

Usage:
    python boundary_marking.py -i aircraft_rib.step --visualize
    python boundary_marking.py -i aircraft_rib.step --export
"""

import gmsh
import meshio
import numpy as np
import argparse
from pathlib import Path



# ============================================================================
# BOUNDARY IDENTIFICATION
# ============================================================================

def identify_boundaries(nodes: np.ndarray, elements: np.ndarray,
                        dirichlet_chord_frac: tuple = (0.8, 1.0),
                        neumann_chord_frac: tuple = (0.0, 0.2),
                        clamp_depth_frac: float = 0.10,
                        neumann_z_tol_frac: float = 0.005,
                        neumann_normal_threshold: float = -0.7,
                        n_profile_bins: int = 200,
                        verbose: bool = True):
    """
    Identify Γ_D, Γ_N, and Γ_free on an aircraft rib mesh.

    Axes: X = chord (longest), Y = thickness (shortest), Z = height.

    Γ_D: Top region of the rib used as a vice clamp. Selected by:
         - Chord position in dirichlet_chord_frac range
         - Z > z_max(x) - clamp_depth_frac * h_local(x)
         where z_max(x) and h_local(x) follow the airfoil envelope.

    Γ_N: Bottom outer surface where load is applied. Selected by:
         - Chord position in neumann_chord_frac range
         - Z within neumann_z_tol_frac * h_local(x) of z_min(x)
         - Triangle normal has downward component (n_z < threshold)
         This detects only the outer skin, not internal features.

    Γ_free: All remaining surfaces.

    Parameters
    ----------
    dirichlet_chord_frac : (lo, hi)
        Chord fraction range for Γ_D. 0.0 = chord start, 1.0 = chord end.
    neumann_chord_frac : (lo, hi)
        Chord fraction range for Γ_N.
    clamp_depth_frac : float
        Γ_D extends this fraction of local height down from local z_max(x).
    neumann_z_tol_frac : float
        Γ_N selects triangles within this fraction of local height from
        the local z_min(x). Default 0.005 (0.5%) restricts to only the
        outermost bottom skin, excluding internal stiffener shelves.
    neumann_normal_threshold : float
        Γ_N requires triangle normal z-component below this value.
        -0.7 selects faces pointing at least 45° downward.
    n_profile_bins : int
        Bins along chord for computing the airfoil envelope.
    """
    extents = nodes.max(axis=0) - nodes.min(axis=0)
    chord_axis = np.argmax(extents)
    thickness_axis = np.argmin(extents)
    height_axis = 3 - chord_axis - thickness_axis

    chord_length = extents[chord_axis]
    height = extents[height_axis]
    chord_min = nodes[:, chord_axis].min()
    chord_max = nodes[:, chord_axis].max()
    z_min_global = nodes[:, height_axis].min()
    z_max_global = nodes[:, height_axis].max()

    if verbose:
        ax_names = ['X', 'Y', 'Z']
        print(f"\n  Rib geometry:")
        print(f"    Chord ({ax_names[chord_axis]}):     "
              f"[{chord_min:.1f}, {chord_max:.1f}] (len: {chord_length:.1f})")
        print(f"    Height ({ax_names[height_axis]}):    "
              f"[{z_min_global:.1f}, {z_max_global:.1f}] (len: {height:.1f})")
        print(f"    Thickness ({ax_names[thickness_axis]}): "
              f"(len: {extents[thickness_axis]:.1f})")

    # ---- Airfoil profile envelope: z_max(x), z_min(x) ----
    bin_edges = np.linspace(chord_min, chord_max, n_profile_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    z_upper_profile = np.full(n_profile_bins, z_min_global)
    z_lower_profile = np.full(n_profile_bins, z_max_global)

    node_chord = nodes[:, chord_axis]
    node_z = nodes[:, height_axis]
    bin_idx = np.digitize(node_chord, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_profile_bins - 1)

    for i in range(n_profile_bins):
        mask = bin_idx == i
        if mask.any():
            z_upper_profile[i] = node_z[mask].max()
            z_lower_profile[i] = node_z[mask].min()

    local_height_profile = z_upper_profile - z_lower_profile

    if verbose:
        print(f"\n  Airfoil envelope (z_max(x), z_min(x)):")
        print(f"    Local height range: [{local_height_profile.min():.1f}, "
              f"{local_height_profile.max():.1f}]")
        sample_fracs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        print(f"    {'Chord%':>8} {'X':>8} {'z_max':>8} {'z_min':>8} {'h_local':>8}")
        for frac in sample_fracs:
            idx = min(int(frac * (n_profile_bins - 1)), n_profile_bins - 1)
            print(f"    {frac:>7.0%} {bin_centers[idx]:>8.1f} "
                  f"{z_upper_profile[idx]:>8.1f} {z_lower_profile[idx]:>8.1f} "
                  f"{local_height_profile[idx]:>8.1f}")

    # ---- Extract surface triangles ----
    surface_tris = _extract_surface_triangles(elements)

    if verbose:
        print(f"\n  Surface triangles: {len(surface_tris):,}")

    # ---- Triangle centroids and normals ----
    v0 = nodes[surface_tris[:, 0]]
    v1 = nodes[surface_tris[:, 1]]
    v2 = nodes[surface_tris[:, 2]]
    tri_centroids = (v0 + v1 + v2) / 3.0
    tri_chord = tri_centroids[:, chord_axis]
    tri_z = tri_centroids[:, height_axis]

    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals / norms
    normal_z = normals[:, height_axis]

    # ---- Local profile values at each triangle's chord position ----
    tri_bin = np.digitize(tri_chord, bin_edges) - 1
    tri_bin = np.clip(tri_bin, 0, n_profile_bins - 1)

    tri_z_max_local = z_upper_profile[tri_bin]
    tri_z_min_local = z_lower_profile[tri_bin]
    tri_h_local = local_height_profile[tri_bin]

    # ---- Chord range masks ----
    # Epsilon padding so frac=0.0 and frac=1.0 include the true extremes
    eps = chord_length * 1e-6
    d_chord_lo, d_chord_hi = dirichlet_chord_frac
    n_chord_lo, n_chord_hi = neumann_chord_frac
    d_x_lo = chord_min + d_chord_lo * chord_length - eps
    d_x_hi = chord_min + d_chord_hi * chord_length + eps
    n_x_lo = chord_min + n_chord_lo * chord_length - eps
    n_x_hi = chord_min + n_chord_hi * chord_length + eps

    in_d_chord = (tri_chord >= d_x_lo) & (tri_chord <= d_x_hi)
    in_n_chord = (tri_chord >= n_x_lo) & (tri_chord <= n_x_hi)

    # ---- Γ_D: top region, depth band from local z_max(x) ----
    tri_z_d_cutoff = tri_z_max_local - clamp_depth_frac * tri_h_local
    dirichlet_mask = in_d_chord & (tri_z > tri_z_d_cutoff)

    # ---- Γ_N: bottom outer surface only ----
    # Proximity to local z_min(x): z < z_min(x) + tol * h_local(x)
    tri_z_n_cutoff = tri_z_min_local + neumann_z_tol_frac * tri_h_local
    near_bottom = tri_z < tri_z_n_cutoff
    # Normal points downward (outer bottom skin, not internal features)
    faces_down = normal_z < neumann_normal_threshold
    neumann_mask = in_n_chord & near_bottom & faces_down

    # ---- Γ_free ----
    free_mask = ~dirichlet_mask & ~neumann_mask

    # ---- Extract sets ----
    dirichlet_tris = surface_tris[dirichlet_mask]
    neumann_tris = surface_tris[neumann_mask]
    free_tris = surface_tris[free_mask]

    dirichlet_nodes = np.unique(dirichlet_tris) if len(dirichlet_tris) > 0 \
        else np.array([], dtype=int)
    neumann_nodes = np.unique(neumann_tris) if len(neumann_tris) > 0 \
        else np.array([], dtype=int)
    free_nodes = np.unique(free_tris) if len(free_tris) > 0 \
        else np.array([], dtype=int)

    if verbose:
        print(f"\n  Γ_D (clamp, top region):")
        print(f"    Chord: X in [{d_x_lo:.1f}, {d_x_hi:.1f}] "
              f"(frac [{d_chord_lo}, {d_chord_hi}])")
        print(f"    Depth: top {clamp_depth_frac:.0%} of local height")
        print(f"    → {len(dirichlet_tris):,} faces, "
              f"{len(dirichlet_nodes):,} nodes")

        print(f"\n  Γ_N (load, bottom surface):")
        print(f"    Chord: X in [{n_x_lo:.1f}, {n_x_hi:.1f}] "
              f"(frac [{n_chord_lo}, {n_chord_hi}])")
        print(f"    Z tolerance: {neumann_z_tol_frac:.1%} of local h "
              f"(≈ {neumann_z_tol_frac * local_height_profile.mean():.2f}mm "
              f"at mid-chord)")
        print(f"    Normal: n_z < {neumann_normal_threshold}")
        print(f"    → {len(neumann_tris):,} faces, "
              f"{len(neumann_nodes):,} nodes")

        print(f"\n  Γ_free: {len(free_tris):,} faces")

        total_bc = len(dirichlet_nodes) + len(neumann_nodes)
        print(f"  BC nodes / total = {total_bc}/{len(nodes)} "
              f"({100*total_bc/len(nodes):.1f}%)")

    return {
        "chord_axis": chord_axis,
        "height_axis": height_axis,
        "thickness_axis": thickness_axis,
        "chord_length": chord_length,
        "height": height,
        "dirichlet_nodes": dirichlet_nodes,
        "neumann_nodes": neumann_nodes,
        "free_nodes": free_nodes,
        "dirichlet_tris": dirichlet_tris,
        "neumann_tris": neumann_tris,
        "free_tris": free_tris,
        "surface_tris": surface_tris,
        "dirichlet_chord_frac": dirichlet_chord_frac,
        "neumann_chord_frac": neumann_chord_frac,
        "clamp_depth_frac": clamp_depth_frac,
        "z_upper_profile": z_upper_profile,
        "z_lower_profile": z_lower_profile,
        "profile_bin_centers": bin_centers,
    }


def _extract_surface_triangles(elements: np.ndarray) -> np.ndarray:
    """Extract boundary triangles from tet mesh."""
    face_count = {}
    face_indices = [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]
    for tet in elements:
        for fi in face_indices:
            f = tuple(sorted(tet[fi]))
            face_count[f] = face_count.get(f, 0) + 1
    return np.array([list(f) for f, c in face_count.items() if c == 1])



# ============================================================================
# GMSH EXPORT WITH PHYSICAL GROUPS
# ============================================================================

def generate_marked_mesh(step_file: str, mesh_size_min: float = 1.0,
                         mesh_size_max: float = 6.0,
                         dirichlet_chord_frac: tuple = (0.8, 1.0),
                         neumann_chord_frac: tuple = (0.0, 0.2),
                         clamp_depth_frac: float = 0.10,
                         neumann_z_tol_frac: float = 0.005,
                         neumann_normal_threshold: float = -0.7,
                         output_prefix: str = None):
    """
    Generate a mesh with Gmsh physical groups marking the boundaries.
    This is the clean way to get boundary markers into FEniCSx / dolfinx.
    """
    step_path = Path(step_file)
    if output_prefix is None:
        output_prefix = step_path.stem + "_marked"

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    try:
        # Import and heal
        gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
        gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
        gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
        gmsh.option.setNumber("Geometry.OCCSewFaces", 1)

        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()

        # Repair if needed
        if len(gmsh.model.getEntities(dim=3)) == 0:
            for tol in [1e-4, 1e-3, 1e-2]:
                try:
                    gmsh.model.occ.healShapes(
                        tolerance=tol, fixDegenerated=True,
                        fixSmallEdges=True, fixSmallFaces=True,
                        sewFaces=True, makeSolids=True)
                    gmsh.model.occ.synchronize()
                    if len(gmsh.model.getEntities(dim=3)) > 0:
                        break
                except Exception:
                    pass

        volumes = gmsh.model.getEntities(dim=3)
        surfaces = gmsh.model.getEntities(dim=2)

        if not volumes:
            raise RuntimeError("No volume found after repair")

        # ---- Identify end faces via bounding boxes ----
        # Find chord axis and range from the volume bounding box
        vol_bb = gmsh.model.getBoundingBox(3, volumes[0][1])
        bb_min = np.array(vol_bb[:3])
        bb_max = np.array(vol_bb[3:])
        extents = bb_max - bb_min
        chord_axis = np.argmax(extents)
        chord_min = bb_min[chord_axis]
        chord_max = bb_max[chord_axis]
        chord_length = chord_max - chord_min

        print(f"\n  Chord axis: {'XYZ'[chord_axis]}, "
              f"range: [{chord_min:.1f}, {chord_max:.1f}], "
              f"length: {chord_length:.1f}")

        # Gmsh physical groups: just tag volume + all surfaces
        # Detailed boundary marking happens post-meshing via identify_boundaries

        # Classify surfaces
        dirichlet_surfaces = []
        neumann_surfaces = []
        free_surfaces = []

        # Fractions are given in chord coordinates [0,1] from chord_min to chord_max.
        d_lo, d_hi = dirichlet_chord_frac
        n_lo, n_hi = neumann_chord_frac
        d_x_lo = chord_min + d_lo * chord_length
        d_x_hi = chord_min + d_hi * chord_length
        n_x_lo = chord_min + n_lo * chord_length
        n_x_hi = chord_min + n_hi * chord_length

        for dim, tag in surfaces:
            sbb = gmsh.model.getBoundingBox(dim, tag)
            s_min = sbb[chord_axis]
            s_max = sbb[chord_axis + 3]
            s_center = (s_min + s_max) / 2

            # A face is an end face if it's thin in the chord direction
            # AND located at the end
            s_extent_chord = s_max - s_min
            is_thin = s_extent_chord < 0.1 * chord_length

            if (d_x_lo <= s_center <= d_x_hi) and is_thin:
                dirichlet_surfaces.append(tag)
            elif (n_x_lo <= s_center <= n_x_hi) and is_thin:
                neumann_surfaces.append(tag)
            else:
                free_surfaces.append(tag)

        print(f"  Gmsh surface classification:")
        print(f"    Γ_D surfaces (spar):     {dirichlet_surfaces}")
        print(f"    Γ_N surfaces (tip):      {neumann_surfaces}")
        print(f"    Γ_free surfaces:         {len(free_surfaces)} surfaces")

        # ---- Physical groups ----
        # Volume
        gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], tag=1)
        gmsh.model.setPhysicalName(3, 1, "volume")

        # Boundaries
        if dirichlet_surfaces:
            gmsh.model.addPhysicalGroup(2, dirichlet_surfaces, tag=2)
            gmsh.model.setPhysicalName(2, 2, "dirichlet_spar")

        if neumann_surfaces:
            gmsh.model.addPhysicalGroup(2, neumann_surfaces, tag=3)
            gmsh.model.setPhysicalName(2, 3, "neumann_tip")

        if free_surfaces:
            gmsh.model.addPhysicalGroup(2, free_surfaces, tag=4)
            gmsh.model.setPhysicalName(2, 4, "free")

        # ---- Mesh settings ----
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 10)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        # ---- Generate ----
        print("\n  Generating mesh...")
        gmsh.model.mesh.generate(3)
        for _ in range(3):
            gmsh.model.mesh.optimize("Laplace2D")
            gmsh.model.mesh.optimize("Relocate3D")
        gmsh.model.mesh.optimize("Netgen")

        # ---- Save ----
        msh_file = f"{output_prefix}.msh"
        gmsh.write(msh_file)
        print(f"  Wrote: {msh_file}")

    finally:
        gmsh.finalize()

    # ---- Convert to formats with boundary markers preserved ----
    mesh = meshio.read(msh_file)

    # Extract tet mesh
    tet_cells = [c for c in mesh.cells if c.type in ("tetra", "tetra10")]
    tri_cells = [c for c in mesh.cells if c.type in ("triangle", "triangle6")]

    if tet_cells:
        nodes = mesh.points
        elements = tet_cells[0].data

        # Save VTU — don't pass cell_data, it has mismatched block counts
        # from Gmsh physical groups
        tet_mesh = meshio.Mesh(points=nodes, cells=tet_cells)
        meshio.write(f"{output_prefix}.vtu", tet_mesh)

        # Save numpy arrays with boundary info
        boundaries = identify_boundaries(
            nodes, elements,
            dirichlet_chord_frac=dirichlet_chord_frac,
            neumann_chord_frac=neumann_chord_frac,
            clamp_depth_frac=clamp_depth_frac,
            neumann_z_tol_frac=neumann_z_tol_frac,
            neumann_normal_threshold=neumann_normal_threshold)

        np.savez(f"{output_prefix}_arrays.npz",
                 **{k: v for k, v in boundaries.items()
                    if isinstance(v, (np.ndarray, int, float))},
                 nodes=nodes, elements=elements)

        print(f"  Wrote: {output_prefix}.vtu")
        print(f"  Wrote: {output_prefix}_arrays.npz")

        return nodes, elements, boundaries

    return None, None, None


# ============================================================================
# MARK EXISTING MESHES (without re-meshing)
# ============================================================================

def mark_existing_mesh(npz_file: str,
                       dirichlet_chord_frac: tuple = (0.8, 1.0),
                       neumann_chord_frac: tuple = (0.0, 0.2),
                       clamp_depth_frac: float = 0.10,
                       neumann_z_tol_frac: float = 0.005,
                       neumann_normal_threshold: float = -0.7):
    """Mark boundaries on an existing mesh from a .npz file."""
    data = np.load(npz_file, allow_pickle=True)
    nodes = data["nodes"]
    elements = data["elements"]

    print(f"\n  Marking boundaries on: {npz_file}")
    print(f"    Nodes: {len(nodes):,}, Tets: {len(elements):,}")

    boundaries = identify_boundaries(
        nodes, elements,
        dirichlet_chord_frac=dirichlet_chord_frac,
        neumann_chord_frac=neumann_chord_frac,
        clamp_depth_frac=clamp_depth_frac,
        neumann_z_tol_frac=neumann_z_tol_frac,
        neumann_normal_threshold=neumann_normal_threshold)

    out_file = str(Path(npz_file).with_stem(
        Path(npz_file).stem.replace("_arrays", "") + "_marked"))
    np.savez(out_file, **{k: v for k, v in boundaries.items()
                          if isinstance(v, (np.ndarray, int, float))},
             nodes=nodes, elements=elements)

    print(f"  Wrote: {out_file}")
    return nodes, elements, boundaries


# ============================================================================
# VISUALIZATION
# ============================================================================

def _dim_label(d):
    return ["X", "Y", "Z"][d]


# ============================================================================
# SUMMARY OF THE WELL-POSED PDE
# ============================================================================

def print_pde_summary(boundaries):
    n_d = len(boundaries["dirichlet_nodes"])
    n_n = len(boundaries["neumann_nodes"])
    chord = boundaries["chord_length"]

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  LINEAR ELASTICITY PROBLEM ON AIRCRAFT RIB                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Strong form:                                                ║
║    -div(σ(u)) = 0          in Ω     (equilibrium)           ║
║                                                              ║
║  where:                                                      ║
║    ε = sym(∇u)             (infinitesimal strain)            ║
║    σ = λ tr(ε) I + 2μ ε    (Hooke)                           ║
║                                                              ║
║  Boundary conditions:                                        ║
║    u = 0         on Γ_D  (upper spar face, {n_d:>5} nodes)       ║
║    σ·n = t_k     on Γ_N  (trailing edge tip, {n_n:>4} nodes)      ║
║    σ·n = 0       on Γ_free (all remaining surfaces)          ║
║                                                              ║
║  Geometry:                                                   ║
║    Γ_D = spar end face, upper portion only (spar cap clamp)  ║
║    Γ_N = trailing edge tip face (aero load transfer)         ║
║    Γ_free = web, flanges, holes, stiffeners, lower spar face ║
║                                                              ║
║  Material:                                                   ║
║    μ(x) = E(x) / (2(1+ν))                                   ║
║    λ(x) = E(x)·ν / ((1+ν)(1-2ν))                            ║
║                                                              ║
║  Input field (for neural operator):                          ║
║    E(x) ~ GP(μ_E, k(x,x'))   clipped to [E_min, E_max]     ║
║                                                              ║
║  Output field:                                               ║
║    u(x) = (u_x, u_y, u_z)    displacement at each node      ║
║                                                              ║
║  Neural operator learns:  G: E(x) ↦ u(x)                    ║
║  Chord length: {chord:.1f}                                         ║
╚══════════════════════════════════════════════════════════════╝
""")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Boundary marking for aircraft rib linear elasticity")
    p.add_argument("--input", "-i", type=str, default=None,
                   help="STEP file (generates new marked mesh)")
    p.add_argument("--npz", type=str, default=None,
                   help="Existing .npz mesh file (mark without re-meshing)")
    p.add_argument("--min-size", type=float, default=1.0)
    p.add_argument("--max-size", type=float, default=6.0)
    p.add_argument("--dirichlet-chord", type=float, nargs=2, default=[0.8, 1.0],
                   metavar=("LO", "HI"),
                   help="Chord fraction range for Γ_D (default: 0.8 1.0)")
    p.add_argument("--neumann-chord", type=float, nargs=2, default=[0.0, 0.2],
                   metavar=("LO", "HI"),
                   help="Chord fraction range for Γ_N (default: 0.0 0.2)")
    p.add_argument("--clamp-depth", type=float, default=0.10,
                   help="Γ_D: fraction of local height from top (default 0.10)")
    p.add_argument("--neumann-z-tol", type=float, default=0.005,
                   help="Γ_N: Z proximity as fraction of local height "
                        "(default 0.005 = 0.5%%)")
    p.add_argument("--neumann-normal", type=float, default=-0.7,
                   help="Γ_N: normal z-component threshold (default -0.7)")
    p.add_argument("--export", action="store_true",
                   help="Export marked mesh")
    p.add_argument("--output-dir", type=str, default=".")

    args = p.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(exist_ok=True)

    if args.npz:
        nodes, elements, boundaries = mark_existing_mesh(
            args.npz,
            dirichlet_chord_frac=tuple(args.dirichlet_chord),
            neumann_chord_frac=tuple(args.neumann_chord),
            clamp_depth_frac=args.clamp_depth,
            neumann_z_tol_frac=args.neumann_z_tol,
            neumann_normal_threshold=args.neumann_normal)
    elif args.input:
        nodes, elements, boundaries = generate_marked_mesh(
            args.input,
            mesh_size_min=args.min_size,
            mesh_size_max=args.max_size,
            dirichlet_chord_frac=tuple(args.dirichlet_chord),
            neumann_chord_frac=tuple(args.neumann_chord),
            clamp_depth_frac=args.clamp_depth,
            neumann_z_tol_frac=args.neumann_z_tol,
            neumann_normal_threshold=args.neumann_normal,
            output_prefix=str(outdir / "rib_marked"))
    else:
        print("Provide either --input (STEP) or --npz (existing mesh)")
        exit(1)

    if boundaries is not None:
        print_pde_summary(boundaries)
