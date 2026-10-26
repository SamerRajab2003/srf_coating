"""
Surface SPDE random field generation and Monte Carlo Q-factor sampling.
Zero CST dependencies. Zero statistical interpretation.
"""
import logging
import time
from typing import Callable, Dict, List, Tuple
import numpy as np
import scipy.sparse as sp

from src.errors import PhysicsError
from src.physics import surface_impedance

logger = logging.getLogger(__name__)


def build_fem_matrices(
    vertices_mm: np.ndarray,
    triangle_node_indices: np.ndarray
) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
    """Assembles surface Mass (M) and Stiffness (K) finite element matrices.

    These matrices discretize the Laplace-Beltrami operator on the cavity shell
    to solve the Stochastic Partial Differential Equation (SPDE) for Gaussian
    random fields with a spatial correlation length.

    Args:
        vertices_mm: Array of vertex coordinates [mm], shape (num_vertices, 3).
        triangle_node_indices: Triangle connectivity, shape (num_triangles, 3).

    Returns:
        Tuple[sp.csc_matrix, sp.csc_matrix]: Mass matrix M_S and Stiffness matrix K_S.
    """
    num_verts = len(vertices_mm)
    num_tris = len(triangle_node_indices)
    v1 = vertices_mm[triangle_node_indices[:, 0]]
    v2 = vertices_mm[triangle_node_indices[:, 1]]
    v3 = vertices_mm[triangle_node_indices[:, 2]]
    e1, e2, e3 = v3 - v2, v1 - v3, v2 - v1
    cross = np.cross(v2 - v1, v3 - v1)
    areas_mm2 = np.maximum(0.5 * np.linalg.norm(cross, axis=1), 1e-12)

    K_data = np.zeros((num_tris, 3, 3))
    M_data = np.zeros((num_tris, 3, 3))
    edges = [e1, e2, e3]
    for i in range(3):
        for j in range(3):
            K_data[:, i, j] = np.sum(
                edges[i] * edges[j], axis=1) / (4.0 * areas_mm2)
            M_data[:, i, j] = areas_mm2 / 12.0 * (2.0 if i == j else 1.0)

    i_inds = triangle_node_indices[:, np.array(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]])]
    j_inds = triangle_node_indices[:, np.array(
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]])]
    K_S = sp.coo_matrix((K_data.flatten(), (i_inds.flatten(
    ), j_inds.flatten())), shape=(num_verts, num_verts)).tocsc()
    M_S = sp.coo_matrix((M_data.flatten(), (i_inds.flatten(
    ), j_inds.flatten())), shape=(num_verts, num_verts)).tocsc()
    isolated = (M_S.diagonal() == 0).astype(float)
    return M_S + sp.diags(isolated, format='csc'), K_S + sp.diags(isolated, format='csc')


def mass_matrix_cholesky(M_S: sp.csc_matrix) -> np.ndarray:
    """Computes the lower-triangular Cholesky factor of the mass matrix.

    Used to transform standard white noise into spatially correlated noise.

    Args:
        M_S: Surface mass matrix.

    Returns:
        np.ndarray: Cholesky factor L such that M_S approx L @ L.T.
    """
    M_dense = M_S.toarray()
    M_dense = (M_dense + M_dense.T) / 2.0 + np.eye(M_S.shape[0]) * 1e-10
    return np.linalg.cholesky(M_dense)


def spde_system_matrix(M_S: sp.csc_matrix, K_S: sp.csc_matrix, length_scale_mm: float) -> sp.csc_matrix:
    """Forms the SPDE system matrix A = (1 / xi^2) * M + K.

    Args:
        M_S: Surface mass matrix.
        K_S: Surface stiffness matrix.
        length_scale_mm: Defect spatial correlation length xi [mm].

    Returns:
        sp.csc_matrix: Sparse system matrix ready for factorization.
    """
    return ((1.0 / length_scale_mm**2) * M_S + K_S).tocsc()


def sample_random_field(solve_A: Callable, L_mass: np.ndarray, length_scale_mm: float, seed: int = None) -> np.ndarray:
    """Generates a single realization of a continuous Gaussian random field on the mesh.

    Args:
        solve_A: Pre-factored linear solver for the SPDE system matrix A.
        L_mass: Cholesky factor of the mass matrix.
        length_scale_mm: Correlation length xi [mm].
        seed: Optional integer seed for reproducibility.

    Returns:
        np.ndarray: Random field values evaluated at each active mesh vertex.
    """
    if seed is not None:
        np.random.seed(seed)
    w = np.random.randn(L_mass.shape[0])
    return solve_A(np.sqrt(8.0 * np.pi / length_scale_mm) * (L_mass @ w))


def sample_one_q(
    solve_A: Callable,
    L_mass: np.ndarray,
    triangle_node_indices: np.ndarray,
    L1: np.ndarray,
    L2: np.ndarray,
    L3: np.ndarray,
    J_squared_quad: np.ndarray,
    areas_m2: np.ndarray,
    quad_weights: np.ndarray,
    freq_hz: float,
    stored_energy_J: float,
    length_scale_mm: float,
    coating_cfg,
    impedance_cfg,
    seed: int = None
) -> Tuple[float, float, float, Dict[str, float]]:
    """Evaluates one random cavity realization and integrates Q, Delta f, and F_hot.

    Args:
        solve_A: Linear solver for the SPDE system.
        L_mass: Mass matrix Cholesky factor.
        triangle_node_indices: Triangle connectivity.
        L1, L2, L3: Barycentric quadrature coordinates.
        J_squared_quad: Quadrature surface current magnitude squared |J|^2 [A^2/m^2].
        areas_m2: Triangle surface areas [m^2].
        quad_weights: Dunavant quadrature weights.
        freq_hz: Unperturbed eigenmode frequency [Hz].
        stored_energy_J: Unperturbed cavity stored energy [J].
        length_scale_mm: Coating correlation length [mm].
        coating_cfg: Coating configuration (mean thickness, max deviation).
        impedance_cfg: Surface impedance model configuration.
        seed: Random seed for this specific sample.

    Returns:
        Tuple containing:
            float: Cavity Quality Factor Q_0 (dimensionless).
            float: Slater frequency shift Delta f [Hz].
            float: Hot-spot quench factor F_hot (dimensionless).
            Dict[str, float]: Execution time breakdown [s].

    Raises:
        PhysicsError: If integrated surface power dissipation is zero or negative.
    """
    timings = {}
    t0 = time.perf_counter()
    raw_field_verts = sample_random_field(
        solve_A, L_mass, length_scale_mm, seed=seed)
    timings["SPDE generation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    valid_field = raw_field_verts[triangle_node_indices]
    field_min, field_max = np.min(valid_field), np.max(valid_field)
    """ max_abs = max(abs(field_min), abs(field_max))
    if max_abs < 1e-12:
        max_abs = 1e-12
    normalized_field = raw_field_verts / max_abs """
    if field_max == field_min:
        field_max = field_min + 1e-12
    normalized_field = 2.0 * (raw_field_verts - field_min) / \
        (field_max - field_min) - 1.0

    mean_m = coating_cfg.mean_thickness_nm * 1e-9
    dev_m = coating_cfg.max_deviation_nm * 1e-9
    d_S_verts = mean_m + dev_m * normalized_field
    d0 = d_S_verts[triangle_node_indices[:, 0]]
    d1 = d_S_verts[triangle_node_indices[:, 1]]
    d2 = d_S_verts[triangle_node_indices[:, 2]]
    d_S_quad = np.outer(d0, L1) + np.outer(d1, L2) + np.outer(d2, L3)
    timings["Thickness interpolation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    Z_local = surface_impedance(d_S_quad.flatten(), freq_hz, impedance_cfg)
    Rs_local_quad = np.real(Z_local).reshape(J_squared_quad.shape)
    Xs_local_quad = np.imag(Z_local).reshape(J_squared_quad.shape)
    timings["Surface impedance"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    areas_expanded = areas_m2[:,
                              np.newaxis] if areas_m2.ndim == 1 else areas_m2
    quad_w = quad_weights[np.newaxis, :]
    local_p = 0.5 * Rs_local_quad * J_squared_quad
    power_loss_W = np.sum(areas_expanded * quad_w * local_p)

    if power_loss_W <= 0.0:
        raise PhysicsError(
            f"Non-physical power loss {power_loss_W:.3e} W at length scale {length_scale_mm:.1f} mm."
        )

    # 1. Quality factor Q_0
    Q = (2.0 * np.pi * freq_hz * stored_energy_J) / power_loss_W
    timings["Power integration"] = time.perf_counter() - t0
    timings["Total"] = sum(timings.values())

    # 2. Slater frequency shift Delta f [Hz]
    reactance_integral = np.sum(
        areas_expanded * quad_w * Xs_local_quad * J_squared_quad)
    delta_f_hz = -reactance_integral / (8.0 * np.pi * stored_energy_J)

    # 3. Hot-spot peakiness factor F_hot
    total_area = np.sum(areas_expanded * quad_w)
    mean_p = power_loss_W / total_area
    f_hot = np.max(local_p) / mean_p

    return Q, delta_f_hz, f_hot, timings


""" def run_monte_carlo_study(
    vertices_mm: np.ndarray,
    triangle_node_indices: np.ndarray,
    L1: np.ndarray,
    L2: np.ndarray,
    L3: np.ndarray,
    J_squared_quad: np.ndarray,
    areas_m2: np.ndarray,
    quad_weights: np.ndarray,
    freq_hz: float,
    stored_energy_J: float,
    cfg
) -> Tuple[Dict[int, dict], List[int], float]: """


def run_monte_carlo_study(
    vertices_mm: np.ndarray,
    triangle_node_indices: np.ndarray,
    L1: np.ndarray,
    L2: np.ndarray,
    L3: np.ndarray,
    J_squared_quad: np.ndarray,
    areas_m2: np.ndarray,
    quad_weights: np.ndarray,
    freq_hz: float,
    stored_energy_J: float,
    cfg
) -> Tuple[Dict[float, dict], List[float], float]:
    """Orchestrates the Monte Carlo sweep across all specified correlation lengths.

    Args:
        vertices_mm: Mesh vertex coordinates [mm].
        triangle_node_indices: Mesh surface triangle connectivity.
        L1, L2, L3: Barycentric coordinates for 7-point Dunavant quadrature.
        J_squared_quad: Surface current magnitude squared on quad points [A^2/m^2].
        areas_m2: Surface triangle areas [m^2].
        quad_weights: Dunavant quadrature weights.
        freq_hz: Cavity resonance frequency [Hz].
        stored_energy_J: Cavity stored energy [J].
        cfg: Configuration model containing coating and statistics options.

    Returns:
        Tuple containing:
            Dict[int, dict]: Results mapped by length scale percentage with keys
                'Q_normalized', 'delta_f_hz', and 'f_hot'.
            List[int]: List of evaluated length scale percentages.
            float: Reference Quality Factor for the uniform unperturbed coating.
    """
    active_nodes = np.unique(triangle_node_indices)
    old_to_new = np.full(len(vertices_mm), -1, dtype=int)
    old_to_new[active_nodes] = np.arange(len(active_nodes))
    vertices_active_mm = vertices_mm[active_nodes]
    tris_active_idx = old_to_new[triangle_node_indices]
    logger.info(
        "Surface SPDE mesh: %d active nodes (reduced from %d total SLIM nodes)",
        len(active_nodes),
        len(vertices_mm)
    )

    mean_m = cfg.coating.mean_thickness_nm * 1e-9
    Z_uniform = surface_impedance(np.array([mean_m]), freq_hz, cfg.impedance)
    Rs_uniform = np.real(Z_uniform)[0]
    areas_expanded = areas_m2[:,
                              np.newaxis] if areas_m2.ndim == 1 else areas_m2
    uniform_P = 0.5 * \
        np.sum(areas_expanded *
               quad_weights[np.newaxis, :] * Rs_uniform * J_squared_quad)
    Q_uniform = (2.0 * np.pi * freq_hz * stored_energy_J) / uniform_P
    logger.info("--- UNIFORM REFERENCE Q: %.12e ---", Q_uniform)

    logger.info("Assembling surface FEM matrices...")
    t0 = time.perf_counter()
    M_S, K_S = build_fem_matrices(vertices_active_mm, tris_active_idx)
    logger.info("SPDE matrix assembly finished in %.6f s",
                time.perf_counter() - t0)

    logger.info("Preparing SPDE mass-matrix Cholesky factor...")
    t0 = time.perf_counter()
    L_mass = mass_matrix_cholesky(M_S)
    logger.info("SPDE Cholesky preparation finished in %.3f seconds.",
                time.perf_counter() - t0)

    results_db = {}
    rng = np.random.default_rng(cfg.statistics.base_seed)
    length_scales_mm = cfg.statistics.length_scales_mm

    for l_mm in length_scales_mm:
        logger.info(
            "\n[Running Length Scale: %.2f mm]", l_mm)
        A_mat = spde_system_matrix(M_S, K_S, l_mm)
        solve_A = sp.linalg.factorized(A_mat)
        q_factors = []
        delta_f_list = []
        f_hot_list = []
        timing_totals = {
            "SPDE generation": 0.0,
            "Thickness interpolation": 0.0,
            "Surface impedance": 0.0,
            "Power integration": 0.0,
            "Total": 0.0
        }

        start_time = time.perf_counter()
        for i in range(cfg.statistics.num_samples):
            seed = int(rng.integers(0, np.iinfo(np.uint32).max))
            q, delta_f, f_hot, sample_timings = sample_one_q(
                solve_A, L_mass, tris_active_idx, L1, L2, L3,
                J_squared_quad, areas_m2, quad_weights,
                freq_hz, stored_energy_J, l_mm,
                cfg.coating, cfg.impedance, seed=seed
            )
            q_factors.append(q)
            delta_f_list.append(delta_f)
            f_hot_list.append(f_hot)
            for k in timing_totals:
                timing_totals[k] += sample_timings[k]
            if (i + 1) % 512 == 0:
                logger.debug("  -> Completed %d/%d samples...",
                             i + 1, cfg.statistics.num_samples)

        elapsed = time.perf_counter() - start_time
        logger.info("  -> Finished %d cavities in %.1f seconds.",
                    cfg.statistics.num_samples, elapsed)
        for k in ["SPDE generation", "Thickness interpolation", "Surface impedance", "Power integration"]:
            logger.info(
                "     %-25s: %.6f s  (%5.1f%%)",
                k,
                timing_totals[k] / cfg.statistics.num_samples,
                100 * timing_totals[k] / timing_totals['Total']
            )

        results_db[l_mm] = {
            "Q_normalized": np.array(q_factors) / Q_uniform,
            "delta_f_hz": np.array(delta_f_list),
            "f_hot": np.array(f_hot_list),
        }

    return results_db, length_scales_mm, Q_uniform
