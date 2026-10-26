"""
physics.py
Pure mathematical engine for SRF coating analysis.
Contains: legacy SIS impedance, general transfer-matrix impedance,
surface quadrature (flat/curved), and Q-factor integration.
Zero CST dependencies.
"""
import logging
import math
import numpy as np
import numpy.typing as npt

from src.config import SisParameters
from src.errors import PhysicsError

logger = logging.getLogger(__name__)

# =====================================================================
# PHYSICAL CONSTANTS
# =====================================================================
MU_0 = 4.0 * math.pi * 1e-7          # Vacuum permeability [H/m]
EPS_0 = 8.8541878128e-12             # Vacuum permittivity [F/m]
MM2_TO_M2 = 1e-6                     # Conversion factor: mm^2 to m^2


# =====================================================================
# 1. SURFACE IMPEDANCE MODELS
# =====================================================================
def sis_surface_impedance(
    d_S_m: np.ndarray,
    freq_hz: float,
    params: SisParameters,
) -> npt.NDArray[np.complex128]:
    """Legacy Kubo SIS impedance (Gobeyn et al., Eqs. 15 & 28-32)."""
    omega = 2.0 * np.pi * freq_hz
    d_S_m = np.maximum(d_S_m, 1e-12)

    lambda_1 = params.lambda_1_0_m * \
        (1.0 - (params.T_op_K / params.Tc_1_K)**4)**(-0.5)
    lambda_2 = params.lambda_2_0_m * \
        (1.0 - (params.T_op_K / params.Tc_2_K)**4)**(-0.5)
    sigma_2 = params.RRR_2 * params.sigma_2_300K_S_per_m

    r = (lambda_2 + params.d_I_m) / lambda_1
    gamma_inv = np.cosh(d_S_m / lambda_1) + r * np.sinh(d_S_m / lambda_1)
    gamma = 1.0 / gamma_inv

    Z_inf_1 = 0.5 * params.sigma_1_S_per_m * MU_0**2 * omega**2 * lambda_1**3
    Z_inf_3 = 0.5 * sigma_2 * MU_0**2 * omega**2 * lambda_2**3

    brace_term = (
        - (1.0 - r**2) * (d_S_m / lambda_1)
        + 0.5 * (1.0 + r**2) * np.sinh(2.0 * d_S_m / lambda_1)
        + r * (np.cosh(2.0 * d_S_m / lambda_1) - 1.0)
    )
    Z_I = gamma**2 * (Z_inf_1 * brace_term + 1j *
                      omega * MU_0 * (1.0 - r**2) * d_S_m)
    Z_II = 1j * omega * gamma**2 * MU_0 * params.d_I_m
    Z_III = gamma**2 * Z_inf_3

    return Z_I + Z_II + Z_III


def _normal_conductivity(layer, T_op_K: float) -> float:
    if layer.sigma_S_per_m is not None:
        return layer.sigma_S_per_m
    if layer.sigma_300K_S_per_m is not None and layer.RRR is not None:
        return layer.RRR * layer.sigma_300K_S_per_m
    raise PhysicsError(f"Layer '{layer.name}': no conductivity specified.")


def layer_wave_number(layer, omega: float, T_op_K: float) -> complex:
    """Calculates the complex propagation constant alpha for a layer [1/m].

    Args:
        layer: LayerConfig object with thickness, conductivity, and Tc.
        omega: Angular RF frequency [rad/s].
        T_op_K: Operational temperature [K].

    Returns:
        complex: Wave propagation constant alpha [1/m], with Im(alpha) > 0.
    """
    if layer.type == "insulator":
        eps = EPS_0 * layer.epsilon_r
        # Dielectric loss conductivity [S/m]
        sigma_n = layer.tan_delta * omega * eps
        inv_lam2 = 0.0
    else:
        eps = EPS_0
        # Normal electron conductivity [S/m]
        sigma_n = _normal_conductivity(layer, T_op_K)
        if layer.type == "superconductor":
            # Two-fluid temperature dependence of penetration depth [m]
            lam = layer.lambda_0_m * (1.0 - (T_op_K / layer.Tc_K)**4)**(-0.5)
            inv_lam2 = 1.0 / (lam**2)
        else:
            inv_lam2 = 0.0
    alpha2 = MU_0 * eps * (omega**2) - inv_lam2 + 1j * omega * MU_0 * sigma_n
    return complex(np.sqrt(alpha2))


def transfer_matrix_surface_impedance(
    d_S_m: np.ndarray,
    freq_hz: float,
    layers: list,
    T_op_K: float,
) -> npt.NDArray[np.complex128]:
    """General multilayer surface impedance, Gobeyn et al. (2026) Eqs. 22-26 & 35.
    Layer 0 thickness is the (array-valued) coating thickness d_S; all deeper
    thicknesses are fixed. Vectorized over N points."""
    omega = 2.0 * np.pi * freq_hz
    shape = np.shape(d_S_m)
    d = np.maximum(np.asarray(d_S_m, dtype=float).ravel(), 1e-12)
    N, M = d.size, len(layers)
    alphas = [layer_wave_number(L, omega, T_op_K) for L in layers]

    th = np.zeros((N, M))
    th[:, 0] = d
    for k in range(1, M):
        th[:, k] = (layers[k].thickness_nm or 0.0) * 1e-9
    zif = np.concatenate(
        [np.zeros((N, 1)), np.cumsum(th, axis=1)[:, :-1]], axis=1)

    C = np.zeros((N, 2), dtype=complex)
    C[:, 0] = 1.0                                   # Eq. 25: B_M = 0
    for i in range(M - 2, -1, -1):                  # Eq. 22 recursion
        beta = alphas[i + 1] / alphas[i]            # Eq. 19 (mu_k = mu_0)
        z = zif[:, i + 1]
        a, b = alphas[i], alphas[i + 1]
        t00 = 0.5 * (1 + beta) * np.exp(1j * z * (-a + b))
        t01 = 0.5 * (1 - beta) * np.exp(1j * z * (-a - b))
        t10 = 0.5 * (1 - beta) * np.exp(1j * z * (a + b))
        t11 = 0.5 * (1 + beta) * np.exp(1j * z * (a - b))
        c0, c1 = C[:, 0], C[:, 1]
        C = np.column_stack((t00 * c0 + t01 * c1, t10 * c0 + t11 * c1))

    Z = (omega * MU_0 / alphas[0]) * (C[:, 0] +
                                      C[:, 1]) / (C[:, 0] - C[:, 1])   # Eq. 35
    # Conjugate Z so that Im(Z) > 0 represents inductive surface reactance (e^{+jwt} convention)
    Z = np.conj(Z)
    return Z.reshape(shape if shape else (1,))


def surface_impedance(
    d_S_m: np.ndarray,
    freq_hz: float,
    impedance_cfg,
) -> npt.NDArray[np.complex128]:
    """Dispatcher: legacy Kubo SIS or general transfer-matrix stack."""
    if impedance_cfg.model_type == "transfer_matrix":
        return transfer_matrix_surface_impedance(
            d_S_m, freq_hz, impedance_cfg.resolved_layers(), impedance_cfg.T_op_K)
    return sis_surface_impedance(d_S_m, freq_hz, impedance_cfg.parameters)


# =====================================================================
# 2. QUADRATURE RULES
# =====================================================================
def dunavant_7pt_rule() -> tuple[np.ndarray, np.ndarray]:
    """Returns barycentric coordinates and weights for the 7-point Dunavant rule."""
    sqrt15 = math.sqrt(15.0)
    a = (6.0 - sqrt15) / 21.0
    b = (6.0 + sqrt15) / 21.0
    w0 = 9.0 / 40.0
    w1 = (155.0 - sqrt15) / 1200.0
    w2 = (155.0 + sqrt15) / 1200.0

    bary_coords = np.array([
        [1/3, 1/3, 1/3],
        [a, a, 1.0 - 2*a],
        [a, 1.0 - 2*a, a],
        [1.0 - 2*a, a, a],
        [b, b, 1.0 - 2*b],
        [b, 1.0 - 2*b, b],
        [1.0 - 2*b, b, b]
    ]).T
    weights = np.array([w0, w1, w1, w1, w2, w2, w2])
    return bary_coords, weights


# =====================================================================
# 3. SURFACE GEOMETRY BUILDERS (Order 1 & Order 2)
# =====================================================================
def build_flat_surface_quadrature(
    vertices_mm: np.ndarray,
    triangle_node_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds flat (Order 1) surface quadrature points, areas, and normals."""
    v0 = vertices_mm[triangle_node_indices[:, 0]]
    v1 = vertices_mm[triangle_node_indices[:, 1]]
    v2 = vertices_mm[triangle_node_indices[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    mag = np.linalg.norm(cross, axis=1)

    flat_areas_m2 = 0.5 * mag * MM2_TO_M2
    flat_normals = cross / mag[:, np.newaxis]

    bary_coords, quad_weights = dunavant_7pt_rule()
    L1, L2, L3 = bary_coords[0], bary_coords[1], bary_coords[2]

    num_tris = len(v0)
    num_q = len(quad_weights)
    quad_pts_mm = np.zeros((num_tris, num_q, 3))

    for i in range(num_q):
        quad_pts_mm[:, i, :] = L1[i]*v0 + L2[i]*v1 + L3[i]*v2

    quad_pts_m = quad_pts_mm * 1e-3
    return quad_pts_m, quad_weights, flat_areas_m2, flat_normals


def build_curved_surface_quadrature(
    vertices_mm: np.ndarray,
    triangle_node_indices: np.ndarray,
    c01_mm: np.ndarray,
    c12_mm: np.ndarray,
    c20_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds curved (Order 2) surface quadrature points, Jacobian areas, normals."""
    v0 = vertices_mm[triangle_node_indices[:, 0]]
    v1 = vertices_mm[triangle_node_indices[:, 1]]
    v2 = vertices_mm[triangle_node_indices[:, 2]]

    bary_coords, quad_weights = dunavant_7pt_rule()
    L1, L2, L3 = bary_coords

    ntri = len(v0)
    nq = len(quad_weights)

    quad_pts_mm = np.zeros((ntri, nq, 3))
    curved_areas_m2 = np.zeros((ntri, nq))
    curved_normals = np.zeros((ntri, nq, 3))

    for p in range(nq):
        l1, l2, l3 = L1[p], L2[p], L3[p]

        quad_pts_mm[:, p, :] = (
            l1*v0 + l2*v1 + l3*v2
            - 2.0 * (l1*l2*c01_mm + l2*l3*c12_mm + l3*l1*c20_mm)
        )

        dr_du = (v1 - v0) - 2.0 * ((l1-l2)*c01_mm + l3*c12_mm - l3*c20_mm)
        dr_dv = (v2 - v0) - 2.0 * (-l2*c01_mm + l2*c12_mm + (l1-l3)*c20_mm)

        cr = np.cross(dr_du, dr_dv)
        jac = np.linalg.norm(cr, axis=1)

        curved_areas_m2[:, p] = 0.5 * jac * MM2_TO_M2
        curved_normals[:, p, :] = cr / jac[:, None]

    quad_pts_m = quad_pts_mm * 1e-3
    return quad_pts_m, quad_weights, curved_areas_m2, curved_normals


# =====================================================================
# 4. CURRENT INTERPOLATION & Q-FACTOR INTEGRATION
# =====================================================================
def interpolate_surface_currents_to_quad_points(
    Jx_nodes: np.ndarray,
    Jy_nodes: np.ndarray,
    Jz_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolates 6-node CST surface currents to the 7 Dunavant quadrature points."""
    if Jx_nodes.shape[1] not in (3, 6):
        raise PhysicsError(
            f"Expected 3 or 6 current nodes per triangle, got {Jx_nodes.shape[1]}. "
            "The CST surface-current export order may have changed.")

    bary_coords, _ = dunavant_7pt_rule()
    L1, L2, L3 = bary_coords[0], bary_coords[1], bary_coords[2]
    num_tris = Jx_nodes.shape[0]
    num_q = len(L1)

    Jx_quad = np.zeros((num_tris, num_q), dtype=complex)
    Jy_quad = np.zeros((num_tris, num_q), dtype=complex)
    Jz_quad = np.zeros((num_tris, num_q), dtype=complex)

    for p in range(num_q):
        l1, l2, l3 = L1[p], L2[p], L3[p]

        N1 = l1 * (2.0 * l1 - 1.0)
        N2 = l2 * (2.0 * l2 - 1.0)
        N3 = l3 * (2.0 * l3 - 1.0)
        N4 = 4.0 * l1 * l2
        N5 = 4.0 * l2 * l3
        N6 = 4.0 * l3 * l1

        Jx_quad[:, p] = N1*Jx_nodes[:, 0] + N2*Jx_nodes[:, 1] + N3*Jx_nodes[:,
                                                                            2] + N5*Jx_nodes[:, 3] + N6*Jx_nodes[:, 4] + N4*Jx_nodes[:, 5]
        Jy_quad[:, p] = N1*Jy_nodes[:, 0] + N2*Jy_nodes[:, 1] + N3*Jy_nodes[:,
                                                                            2] + N5*Jy_nodes[:, 3] + N6*Jy_nodes[:, 4] + N4*Jy_nodes[:, 5]
        Jz_quad[:, p] = N1*Jz_nodes[:, 0] + N2*Jz_nodes[:, 1] + N3*Jz_nodes[:,
                                                                            2] + N5*Jz_nodes[:, 3] + N6*Jz_nodes[:, 4] + N4*Jz_nodes[:, 5]

    return Jx_quad, Jy_quad, Jz_quad


def compute_q_from_surface_currents(
    Jx_quad: np.ndarray,
    Jy_quad: np.ndarray,
    Jz_quad: np.ndarray,
    Rs_local_quad: np.ndarray,
    quad_weights: np.ndarray,
    areas_m2: np.ndarray,
    freq_hz: float,
    stored_energy_J: float,
) -> float:
    """Integrates surface power loss and calculates the Quality Factor Q."""
    J_squared_quad = np.real(
        Jx_quad * np.conj(Jx_quad) +
        Jy_quad * np.conj(Jy_quad) +
        Jz_quad * np.conj(Jz_quad)
    )

    if areas_m2.ndim == 1:
        areas_expanded = areas_m2[:, np.newaxis]
    else:
        areas_expanded = areas_m2

    power_loss_W = 0.5 * np.sum(
        areas_expanded * quad_weights[np.newaxis,
                                      :] * Rs_local_quad * J_squared_quad
    )

    omega = 2.0 * math.pi * freq_hz
    return (omega * stored_energy_J) / power_loss_W


def run_sanity_checks(quad_weights: np.ndarray, areas_m2: np.ndarray, J_squared_quad: np.ndarray):
    """Lightweight validation to prevent silent physics failures."""
    w_sum = float(np.sum(quad_weights))
    if abs(w_sum - 1.0) >= 1e-10:
        raise PhysicsError(
            f"Quadrature weights must sum to 1.0, got {w_sum:.12f}.")
    n_bad = int(np.sum(areas_m2 <= 0.0))
    if n_bad:
        raise PhysicsError(
            f"{n_bad} surface area element(s) <= 0; the mesh is degenerate.")
    if float(np.sum(J_squared_quad)) <= 0.0:
        raise PhysicsError(
            "Surface currents are all zero. Check the CST extraction "
            "(mode number, evaluated Surface Current result).")
    if areas_m2.ndim == 2:
        total_area = float(np.sum(areas_m2 * quad_weights[np.newaxis, :]))
    else:
        total_area = float(np.sum(areas_m2))
    logger.info(
        "    Sanity check passed. Total active area: %.6f m^2", total_area)


def compute_field_metrics_from_raw(
    vertices_mm: np.ndarray,
    triangle_node_indices: np.ndarray,
    Jx_nodes: np.ndarray,
    Jy_nodes: np.ndarray,
    Jz_nodes: np.ndarray,
) -> tuple[float, float]:
    """Calculates Effective Loss Area (A_eff), Dimensional Peakiness (1/sqrt(A_eff)),
        and Dimensionless Shape Peakiness Pi = sqrt(A_total / A_eff).
        """
    J2 = np.real(Jx_nodes * np.conj(Jx_nodes) + Jy_nodes *
                 np.conj(Jy_nodes) + Jz_nodes * np.conj(Jz_nodes)).mean(axis=1)
    verts_m = vertices_mm * 1e-3
    v0 = verts_m[triangle_node_indices[:, 0]]
    v1 = verts_m[triangle_node_indices[:, 1]]
    v2 = verts_m[triangle_node_indices[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    A_total = float(np.sum(areas))
    int_J2 = float(np.sum(areas * J2))
    int_J4 = float(np.sum(areas * (J2**2)))
    A_eff = (int_J2**2) / int_J4
    A_eff_normalized = A_eff / A_total
    peakiness = float(np.sqrt(A_eff_normalized))
    return A_eff_normalized, peakiness
