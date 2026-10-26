"""
Master pipeline orchestrating CST extraction, physics integration, and statistical Monte Carlo.
Outputs RAW DATA ONLY. Statistical analysis is handled separately by analyze.py.
Usage: python run.py <config_file.yaml>
"""
import sys
from pathlib import Path
import numpy as np
import logging

from src.logging_setup import setup_logging, attach_file_handler
from src.errors import SrfCoatingError, MeshParseError
from src.config import load_config
from src.cst_reader import (
    connect_to_cst_project, read_mode_data, read_surface_currents,
    read_slim_mesh, extract_curved_edge_vectors
)
from src.physics import (
    build_flat_surface_quadrature, build_curved_surface_quadrature,
    interpolate_surface_currents_to_quad_points, compute_q_from_surface_currents,
    surface_impedance, dunavant_7pt_rule, run_sanity_checks
)
from src.statistics import run_monte_carlo_study
from src.data_manager import (
    create_run_directory, save_config_snapshot,
    save_extracted_fields, save_raw_samples
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    setup_logging(verbose="--verbose" in argv)
    args = [a for a in argv[1:] if not a.startswith("--")]

    if len(args) != 1:
        logger.error(
            "Usage: python run.py <config_file.yaml>  (got %d arguments)", len(args))
        return 2

    try:
        cfg = load_config(Path(args[0]))

        run_dir = create_run_directory(cfg.output.directory)
        attach_file_handler(run_dir)
        save_config_snapshot(cfg.model_dump(), run_dir)
        logger.info("[+] Run directory created: %s", run_dir)

        logger.info("[Phase 1] Extracting data from CST...")
        # New: reads whatever path is written in your YAML config:
        mws = connect_to_cst_project(cfg.project.cst_project_file)
        frequency_hz, stored_energy_J = read_mode_data(
            mws, cfg.project.mode_number)
        logger.info("    Frequency: %.6f GHz | Stored energy: %.6e J",
                    frequency_hz / 1e9, stored_energy_J)

        Jx_1d, Jy_1d, Jz_1d = read_surface_currents(
            mws, cfg.project.mode_number)
        logger.info("    Surface currents extracted.")

        vertices_mm, triangle_node_indices, raw_slim = read_slim_mesh(
            cfg.project.slim_file)
        logger.info("    Mesh parsed: %d vertices, %d triangles.",
                    len(vertices_mm), len(triangle_node_indices))

        c01_mm, c12_mm, c20_mm = extract_curved_edge_vectors(
            raw_slim, triangle_node_indices)
        is_curved_mesh = "curvedelements_edgeorder" in raw_slim

        num_tris = len(triangle_node_indices)
        if len(Jx_1d) % num_tris != 0:
            raise MeshParseError(
                f"Surface current count ({len(Jx_1d)}) is not a multiple of triangle count ({num_tris}). "
                "CST mesh and 3D field results are out of sync. Re-solve the eigenmode in CST."
            )
        samples_per_tri = len(Jx_1d) // num_tris
        Jx_nodes = Jx_1d.reshape((num_tris, samples_per_tri))
        Jy_nodes = Jy_1d.reshape((num_tris, samples_per_tri))
        Jz_nodes = Jz_1d.reshape((num_tris, samples_per_tri))

        save_extracted_fields(run_dir, vertices_mm, triangle_node_indices,
                              Jx_nodes, Jy_nodes, Jz_nodes, frequency_hz, stored_energy_J)
        logger.info("    Raw fields saved to disk.")

        logger.info(
            "[Phase 2] Building surface quadrature and interpolating currents...")
        metric = cfg.surface.metric
        if getattr(cfg.surface, "cst_q_post_processing", False):
            metric = "flat"

        if metric == "flat" or (metric == "auto" and not is_curved_mesh):
            quad_pts_m, quad_weights, areas_m2, normals = build_flat_surface_quadrature(
                vertices_mm, triangle_node_indices)
            logger.info("    Using FLAT surface metric.")
        else:
            quad_pts_m, quad_weights, areas_m2, normals = build_curved_surface_quadrature(
                vertices_mm, triangle_node_indices, c01_mm, c12_mm, c20_mm)
            logger.info("    Using CURVED surface metric.")

        Jx_quad, Jy_quad, Jz_quad = interpolate_surface_currents_to_quad_points(
            Jx_nodes, Jy_nodes, Jz_nodes)

        if getattr(cfg.surface, "cst_q_post_processing", False):
            nrm = normals if normals.ndim == 3 else normals[:, None, :]
            nx, ny, nz = nrm[:, :, 0], nrm[:, :, 1], nrm[:, :, 2]
            J_dot_n = Jx_quad * nx + Jy_quad * ny + Jz_quad * nz
            Jx_quad = Jx_quad - J_dot_n * nx
            Jy_quad = Jy_quad - J_dot_n * ny
            Jz_quad = Jz_quad - J_dot_n * nz
            logger.info(
                "    cst_q_post_processing ON: tangential projection applied.")

        J_squared_quad = np.real(Jx_quad * np.conj(Jx_quad)
                                 + Jy_quad * np.conj(Jy_quad)
                                 + Jz_quad * np.conj(Jz_quad))
        run_sanity_checks(quad_weights, areas_m2, J_squared_quad)

        mean_m = cfg.coating.mean_thickness_nm * 1e-9
        Z_uniform = surface_impedance(
            np.array([mean_m]), frequency_hz, cfg.impedance)
        Rs_uniform = np.real(Z_uniform)[0]

        Q_uniform_check = compute_q_from_surface_currents(
            Jx_quad, Jy_quad, Jz_quad, np.full_like(
                J_squared_quad, Rs_uniform),
            quad_weights, areas_m2, frequency_hz, stored_energy_J)
        logger.info("    Uniform Q sanity check: %.6e", Q_uniform_check)

        logger.info("[Phase 3] Running Monte Carlo statistical study...")
        bary, _ = dunavant_7pt_rule()
        L1, L2, L3 = bary[0], bary[1], bary[2]

        results_db, length_scales_mm, Q_uniform = run_monte_carlo_study(
            vertices_mm, triangle_node_indices, L1, L2, L3,
            J_squared_quad, areas_m2, quad_weights,
            frequency_hz, stored_energy_J, cfg)

        logger.info("[Phase 4] Saving raw data...")
        save_raw_samples(results_db, length_scales_mm, Q_uniform, run_dir)

        logger.info(
            "[SUCCESS] Raw data generation complete. Run 'python analyze.py %s' for statistical interpretation.", run_dir)
        return 0

    except SrfCoatingError as e:
        logger.error("%s", e)
        logger.debug("Traceback:", exc_info=True)
        return 1
    except Exception as e:
        logger.error("Unexpected failure: %s: %s", type(e).__name__, e)
        logger.debug("Traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
