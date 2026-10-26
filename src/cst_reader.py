"""
Read-only extraction of mesh, fields, and mode data from an active CST Studio Suite project.
"""
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import win32com.client
from src.errors import CstConnectionError, CstExtractionError, MeshParseError

logger = logging.getLogger(__name__)


def _flag_com_methods(obj, methods: list):
    if hasattr(obj, "_FlagAsMethod"):
        for m in methods:
            obj._FlagAsMethod(m)


def connect_to_cst_project(cst_project_path: Path):
    """
    Attaches directly to the CST project specified by cst_project_path.
    If the project is already open in CST, it attaches to it.
    If it is closed, it opens it directly from disk.
    """
    cst_project_path = Path(cst_project_path).resolve()
    if not cst_project_path.is_file():
        raise CstConnectionError(
            f"CST project file does not exist: {cst_project_path}")

    try:
        cst_app = win32com.client.Dispatch("CSTStudio.Application")
    except Exception as e:
        raise CstConnectionError(
            "Could not connect to CST Studio Suite. Is CST installed?") from e

    # 1. Try to open the explicit project path
    try:
        mws = cst_app.OpenFile(str(cst_project_path))
        if mws is not None:
            logger.info("Connected to project from path: %s",
                        cst_project_path.name)
            return mws
    except Exception as e:
        logger.debug("OpenFile failed or project already open: %s", e)

    # 2. Fallback: Check active document if already open
    try:
        mws = cst_app.Active3D()
        if mws is not None:
            active_file = Path(mws.GetProjectPath("Root")).resolve()
            if active_file.stem.lower() == cst_project_path.stem.lower():
                logger.info("Attached to already open project: %s",
                            active_file.name)
                return mws
    except Exception:
        pass

    # Clear error message advising the user to close the tab/window
    raise CstConnectionError(
        f"Failed to attach to '{cst_project_path.name}'.\n"
        f"--> TROUBLESHOOTING TIP: If '{cst_project_path.name}' is currently open in CST, "
        f"please close its project tab (or close CST Studio Suite entirely) and rerun.\n"
    )


def read_mode_data(mws, mode_number: int, dummy_rs_ohm: float = 10e-6) -> Tuple[float, float]:
    solver = mws.EigenmodeSolver
    _flag_com_methods(solver, ["GetModeFrequencyInHz"])
    try:
        frequency_hz = float(solver.GetModeFrequencyInHz(mode_number))
    except Exception as e:
        raise CstExtractionError(
            f"Failed to get frequency for mode {mode_number}.") from e

    MU_0 = 4.0 * np.pi * 1e-7
    sigma = (np.pi * frequency_hz * MU_0) / (dummy_rs_ohm**2)
    sigma_str = f"{sigma:.15e}"

    qf = mws.QFactor
    _flag_com_methods(qf, ["Reset", "UseNewMethod", "SetHField",
                      "SetConductivity", "Calculate", "GetTotalEnergy"])
    qf.Reset()
    qf.UseNewMethod(True)
    qf.SetHField(f"Mode {mode_number}")
    for mat in ["**Cond. Enclosure**", "PEC"]:
        try:
            qf.SetConductivity(mat, sigma_str)
        except Exception:
            pass
    qf.Calculate()

    stored_energy_J = float(qf.GetTotalEnergy())
    return frequency_hz, stored_energy_J


def read_surface_currents(mws, mode_number: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    res_tree = mws.ResultTree
    _flag_com_methods(res_tree, ["GetFileFromTreeItem"])

    tree_path = f"2D/3D Results\\Modes\\Mode {mode_number}\\Surface Current"
    file_path = res_tree.GetFileFromTreeItem(tree_path)
    if not file_path:
        raise CstExtractionError(
            f"CST result tree item not found: '{tree_path}'. Evaluate the Surface Current "
            f"result for mode {mode_number} in CST before running the pipeline.")

    try:
        dispid = mws._oleobj_.GetIDsOfNames(0, "Result3D")
        raw_obj = mws._oleobj_.Invoke(dispid, 0, 1, 1, file_path)
        res3d = win32com.client.Dispatch(raw_obj)
    except Exception:
        _flag_com_methods(mws, ["Result3D"])
        res3d = mws.Result3D(file_path)

    if res3d is None:
        raise CstExtractionError("CST returned a null pointer for Result3D.")

    _flag_com_methods(res3d, ["GetArray", "GetLength"])

    xre = np.array(res3d.GetArray("xre"))
    yre = np.array(res3d.GetArray("yre"))
    zre = np.array(res3d.GetArray("zre"))
    xim = np.array(res3d.GetArray("xim"))
    yim = np.array(res3d.GetArray("yim"))
    zim = np.array(res3d.GetArray("zim"))

    return xre + 1j * xim, yre + 1j * yim, zre + 1j * zim


def read_slim_mesh(slim_path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    if not slim_path.is_file():
        raise MeshParseError(
            f"SLIM mesh file not found: {slim_path}. Run the eigenmode solver once, "
            "or set project.slim_file to the correct 3d.slim path.")

    type_map = {
        'float64': '<f8', 'int32': '<i4', 'uint32': '<u4',
        'uint64': '<u8', 'uint8': '<u1', 'int8': '<i1'
    }
    elements = []
    current_el = None

    with open(slim_path, 'rb') as f:
        while True:
            line = f.readline().decode('utf-8').strip()
            if line == 'end':
                break
            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'element' and parts[1] == 'list':
                elements.append({'type': 'list', 'name': parts[3], 'count': int(
                    parts[4]), 'dtype': type_map[parts[2]]})
                current_el = None
            elif parts[0] == 'element' and parts[1] != 'list':
                current_el = {'type': 'struct', 'name': parts[1], 'count': int(
                    parts[2]), 'props': []}
                elements.append(current_el)
            elif parts[0] == 'EdgeCurvatureDataSets':
                current_el = {'type': 'struct', 'name': parts[0], 'count': int(
                    parts[1]), 'props': []}
                elements.append(current_el)
            elif parts[0] == 'property':
                if current_el is not None:
                    prop_name = parts[2]
                    prop_type = type_map[parts[1]]
                    if not any(p[0] == prop_name for p in current_el['props']):
                        current_el['props'].append((prop_name, prop_type))
            else:
                if parts[0] not in ['format', 'curvature']:
                    current_el = None

        f.seek(f.tell())
        data = {}
        for el in elements:
            if el['count'] == 0:
                continue
            if el['type'] == 'list':
                data[el['name']] = np.fromfile(
                    f, dtype=el['dtype'], count=el['count'])
            elif el['type'] == 'struct':
                dt = np.dtype(el['props'])
                data[el['name']] = np.fromfile(f, dtype=dt, count=el['count'])

    if 'node' not in data or 'triangle' not in data:
        raise MeshParseError(
            "SLIM file is missing 'node' or 'triangle' elements.")

    nodes = data['node']
    vertices_mm = np.column_stack((nodes['x'], nodes['y'], nodes['z']))
    tris = data['triangle']
    triangle_node_indices = np.column_stack(
        (tris['node_index0'], tris['node_index1'], tris['node_index2']))

    return vertices_mm, triangle_node_indices, data


def extract_curved_edge_vectors(raw_slim_data: dict, triangle_node_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_tris = len(triangle_node_indices)
    if 'curvedelements_edgeorder' not in raw_slim_data:
        return np.zeros((n_tris, 3)), np.zeros((n_tris, 3)), np.zeros((n_tris, 3))

    coeff_idx = raw_slim_data['curvedelements_edgecoeffindices']
    coeffs = raw_slim_data['curvedelements_edgecoeffs'].reshape(-1, 3)
    edge_sets = raw_slim_data['EdgeCurvatureDataSets']

    curved_edges = {}
    for row in edge_sets:
        n0 = int(row['node_index0'])
        n1 = int(row['node_index1'])
        edge_id = int(row['dataSet_idx'])
        start = int(coeff_idx[edge_id])
        end = int(coeff_idx[edge_id + 1])
        if end > start:
            c = coeffs[start]
            curved_edges[tuple(sorted((n0, n1)))] = c

    def get_c(a, b):
        return curved_edges.get(tuple(sorted((int(a), int(b)))), np.zeros(3))

    i0 = triangle_node_indices[:, 0]
    i1 = triangle_node_indices[:, 1]
    i2 = triangle_node_indices[:, 2]

    c01_mm = np.array([get_c(a, b) for a, b in zip(i0, i1)])
    c12_mm = np.array([get_c(a, b) for a, b in zip(i1, i2)])
    c20_mm = np.array([get_c(a, b) for a, b in zip(i2, i0)])

    return c01_mm, c12_mm, c20_mm
