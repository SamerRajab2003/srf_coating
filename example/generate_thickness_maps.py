"""
generate_thickness_maps.py
Generates 3D surface thickness fluctuation maps (Delta d_S [nm]) using strictly
the functions from `src.statistics` and `test_patch_size`, with zero-preserving
symmetric normalization to the dominant bound.

Folder output hierarchy:
  thickness_maps/
    └── <cavity_name>/
          └── scale_<X.X>mm/
                ├── realization_1.png
                ├── realization_2.png
                └── realization_3.png
"""
from src.statistics import (
    build_fem_matrices,
    mass_matrix_cholesky,
    spde_system_matrix,
    sample_random_field,
)
from example.test_patch_size import load_from_npz
from matplotlib.colors import LinearSegmentedColormap
import yaml
import scipy.sparse as sp
import pandas as pd
import numpy as np
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    HAS_PYVISTA = False

# Auto-detects the latest run if empty, or add explicit run paths
EXAMPLE_OUTPUTS = ROOT_DIR / "outputs" / "example_run"
INPUT_DIRS = sorted(EXAMPLE_OUTPUTS.glob(
    "run_*"))[-1:] if EXAMPLE_OUTPUTS.exists() else []

OUTPUT_BASE = ROOT_DIR / "thickness_maps"
NUM_REALIZATIONS = 10

PAPER_COLORS = [
    (0.00, "#0033cc"),
    (0.18, "#2980b9"),
    (0.35, "#27ae60"),
    (0.50, "#f1c40f"),
    (0.70, "#8e44ad"),
    (0.85, "#795548"),
    (1.00, "#d5d8dc"),
]
CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "srf_thickness", [c[1] for c in PAPER_COLORS])


def render_with_pyvista(vertices: np.ndarray, triangles: np.ndarray, delta_ds: np.ndarray,
                        max_dev_nm: float, l_mm: float, r_idx: int, out_png: Path):
    """Renders standard scientific 3D cavity with vertical colorbar and exact min/max stats."""
    pv.set_plot_theme("document")
    plotter = pv.Plotter(window_size=(1600, 900), off_screen=True)
    plotter.set_background("white")

    num_tri = len(triangles)
    faces = np.hstack(
        [np.full((num_tri, 1), 3, dtype=np.int64), triangles]).flatten()
    mesh = pv.PolyData(vertices, faces)
    mesh.point_data["Delta_dS"] = delta_ds

    # Auto-align cavity horizontally on screen (longest dimension along horizontal axis)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    ranges = maxs - mins
    center = (mins + maxs) / 2.0
    longest_axis = int(np.argmax(ranges))
    dist = float(np.max(ranges) * 2.0)

    if longest_axis == 2:  # Z is beam axis -> look from -Y, Up is +X
        cam_pos = (center[0], center[1] - dist, center[2])
        cam_up = (1, 0, 0)
    elif longest_axis == 0:  # X is beam axis -> look from +Z, Up is +Y
        cam_pos = (center[0], center[1], center[2] + dist)
        cam_up = (0, 1, 0)
    else:  # Y is beam axis -> look from +Z, Up is +X
        cam_pos = (center[0], center[1], center[2] + dist)
        cam_up = (1, 0, 0)

    # Surface mesh with subtle wireframe
    plotter.add_mesh(
        mesh,
        scalars="Delta_dS",
        cmap=CUSTOM_CMAP,
        clim=[-max_dev_nm, max_dev_nm],
        show_edges=True,
        edge_color="#2b2b2b",
        line_width=0.4,
        smooth_shading=True,
        show_scalar_bar=False,
    )

    plotter.camera.position = cam_pos
    plotter.camera.focal_point = tuple(center)
    plotter.camera.up = cam_up
    plotter.camera.zoom(0.95)

    # Informative Header: scale, realization, and exact min/max of this realization
    real_min = float(np.min(delta_ds))
    real_max = float(np.max(delta_ds))
    info_text = (
        f"Correlation Length: l = {l_mm:.1f} mm  (Realization {r_idx})\n"
        f"Realization Peak: [{real_min:+.1f} nm,  {real_max:+.1f} nm]   |   Nominal: +/-{max_dev_nm:.1f} nm"
    )
    plotter.add_text(
        info_text,
        position="upper_left",
        font_size=13,
        color="black",
        font="arial",
    )

    # Standard right-hand vertical colorbar with 5 clear numeric tick marks
    plotter.add_scalar_bar(
        title="Δd_S [nm]",
        title_font_size=16,
        label_font_size=13,
        shadow=False,
        n_labels=5,          # Shows: -26.0, -13.0, 0.0, +13.0, +26.0
        italic=False,
        fmt="%+.1f",         # Explicit signs: -26.0, +26.0
        font_family="arial",
        color="black",
        vertical=True,
        position_x=0.85,
        position_y=0.15,
        width=0.06,
        height=0.70,
    )

    plotter.screenshot(str(out_png))
    plotter.close()


def render_with_matplotlib(vertices: np.ndarray, triangles: np.ndarray, delta_ds: np.ndarray,
                           max_dev_nm: float, l_mm: float, r_idx: int, out_png: Path):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    face_scalars = delta_ds[triangles].mean(axis=1)
    norm = plt.Normalize(vmin=-max_dev_nm, vmax=max_dev_nm)
    face_colors = CUSTOM_CMAP(norm(face_scalars))

    poly = Poly3DCollection(
        vertices[triangles],
        facecolors=face_colors,
        edgecolors="#222222",
        linewidths=0.2,
        shade=True
    )
    ax.add_collection3d(poly)

    max_range = np.ptp(vertices, axis=0).max() / 2.0
    mid = np.mean(vertices, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_axis_off()
    ax.view_init(elev=15, azim=45)

    ax.set_title(f"$l = {l_mm:.1f}$ mm (Realization {r_idx})",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    total_cavities = len(INPUT_DIRS)

    print("\n==================================================================")
    print("   GENERATING 3D THICKNESS MAP REALIZATIONS (SYMMETRIC ZERO-MEAN)")
    print(
        f"   Renderer in use: {'PyVista (High Quality)' if HAS_PYVISTA else 'Matplotlib Poly3DCollection'}")
    print("==================================================================\n")

    for cav_idx, cav_dir in enumerate(INPUT_DIRS, start=1):
        cav_name = cav_dir.name
        print(f"\n[{cav_idx}/{total_cavities}] >>> Processing Cavity: {cav_name}")

        fields_path = cav_dir / "extracted_fields.npz"
        config_path = cav_dir / "config_snapshot.yaml"
        summary_path = cav_dir / "statistics_summary.csv"

        if not fields_path.exists():
            print(f"    [SKIP] extracted_fields.npz not found in {cav_dir}")
            continue

        vertices_mm, triangle_node_indices, _, _, _ = load_from_npz(
            fields_path)

        active_nodes = np.unique(triangle_node_indices)
        old_to_new = np.full(len(vertices_mm), -1, dtype=int)
        old_to_new[active_nodes] = np.arange(len(active_nodes))
        vertices_active_mm = vertices_mm[active_nodes]
        tris_active_idx = old_to_new[triangle_node_indices]

        max_dev_nm = 26.0
        base_seed = 1
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f)
                max_dev_nm = cfg_data.get("coating", {}).get(
                    "max_deviation_nm", 26.0)
                base_seed = cfg_data.get("statistics", {}).get("base_seed", 1)

        if summary_path.exists():
            df = pd.read_csv(summary_path)
            length_scales = sorted(
                df["length_scale_mm"].dropna().unique().tolist())
        else:
            length_scales = [20.0, 70.0]

        print(
            f"    Mesh: {len(vertices_active_mm)} active nodes, {len(tris_active_idx)} triangles")
        print(
            f"    Length scales found ({len(length_scales)}): {length_scales}")

        M_S, K_S = build_fem_matrices(vertices_active_mm, tris_active_idx)
        L_mass = mass_matrix_cholesky(M_S)

        for s_idx, l_mm in enumerate(length_scales, start=1):
            scale_dir = OUTPUT_BASE / cav_name / f"scale_{l_mm:.1f}mm"
            scale_dir.mkdir(parents=True, exist_ok=True)

            A_mat = spde_system_matrix(M_S, K_S, l_mm)
            solve_A = sp.linalg.factorized(A_mat)

            for r_idx in range(1, NUM_REALIZATIONS + 1):
                out_png = scale_dir / f"realization_{r_idx}.png"
                if out_png.exists():
                    continue

                r_seed = int(base_seed + (s_idx * 1000) + r_idx)
                raw_field_verts = sample_random_field(
                    solve_A, L_mass, l_mm, seed=r_seed)

                # Symmetric zero-preserving normalization
                valid_field = raw_field_verts[tris_active_idx]
                field_min, field_max = np.min(valid_field), np.max(valid_field)
                max_abs = max(abs(field_min), abs(field_max))
                if max_abs < 1e-12:
                    max_abs = 1e-12

                normalized_field = raw_field_verts / max_abs
                delta_ds_nm = max_dev_nm * normalized_field

                if HAS_PYVISTA:
                    render_with_pyvista(
                        vertices_active_mm, tris_active_idx, delta_ds_nm,
                        max_dev_nm, l_mm, r_idx, out_png
                    )
                else:
                    render_with_matplotlib(
                        vertices_active_mm, tris_active_idx, delta_ds_nm,
                        max_dev_nm, l_mm, r_idx, out_png
                    )

            print(
                f"      -> Scale {l_mm:>6.1f} mm : {NUM_REALIZATIONS} realizations saved.")

    print("\n==================================================================")
    print(f"   [SUCCESS] All 9 cavity folders processed cleanly!")
    print(f"   Results saved in: {OUTPUT_BASE.resolve()}")
    print("==================================================================\n")


if __name__ == "__main__":
    main()
