"""
CST Cavity Setup and Simulation Configuration.
Self-contained module: generates TESLA 9-cell geometry, applies boundaries,
sets background material (PEC), configures the HF mesh (order-2 curvature)
and the AKS eigenmode solver via VBA injection.

Fix vs. previous version: the equator arc (S5) is now built as a circle of
radius Aeq and scaled in z by Beq/Aeq (same trick as the iris arc S3), so
Aeq != Beq gives a properly closed profile.
"""
from pathlib import Path
from scipy.optimize import root
import numpy as np
import logging
import sys
import os

# --- CST environment bootstrap (MUST run before importing cst.interface) ---
_CST_AMD64 = r"C:\Program Files (x86)\CST Studio Suite 2025\AMD64"
_CST_PYLIB = os.path.join(_CST_AMD64, "python_cst_libraries")
if _CST_PYLIB not in sys.path:
    sys.path.append(_CST_PYLIB)
os.environ['PATH'] = _CST_AMD64 + ";" + os.environ.get('PATH', '')
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(_CST_AMD64)
# ---------------------------------------------------------------------------


logger = logging.getLogger(__name__)

# =====================================================================
# 1. CONFIGURATION (Edit these to change the cavity or solver)
# =====================================================================

# --- GEOMETRY: CUPS (Wanzenberg 2001) ---
MID_CUP = {
    'name': 'Mid_Cup', 'L': 27.7,
    'Req': 103.3, 'Aeq': 42.0, 'Beq': 12.0,
    'Rir': 35.0,  'Air': 19.0, 'Bir': 12.0
}
END_CUP_1 = {
    'name': 'End_Cup_1', 'L': 56.0,
    'Req': 103.3, 'Aeq': 40.3, 'Beq': 40.3,
    'Rir': 39.0,  'Air': 10.0, 'Bir': 13.5
}
END_CUP_2 = {
    'name': 'End_Cup_2', 'L': 57.0,
    'Req': 103.3, 'Aeq': 42.0, 'Beq': 42.0,
    'Rir': 39.0,  'Air': 9.0,  'Bir': 12.8
}

# --- GEOMETRY: PIPES ---
PIPE_CONFIG = {'length': 105.0, 'radius': 35.0, 'thickness': 2.0}

# --- CAVITY LAYOUT ---
CAVITY = {
    'num_cells': 1,
    'end_left':  'MID_CUP',
    'end_right': 'MID_CUP',
}

# --- SIMULATION & BOUNDARIES ---
SIMULATION_SETUP = {
    'solver_type': 'HF Eigenmode',
    'frequency_range': [1.26, 1.32],  # GHz
    'boundaries': {
        'xmin': 'electric', 'xmax': 'electric',
        'ymin': 'electric', 'ymax': 'electric',
        'zmin': 'electric', 'zmax': 'electric'
    },
    'symmetry': {'x': 'none', 'y': 'none', 'z': 'none'},
    'background_material': 'Pec',   # kept as PEC on purpose
}

# --- MESH SETTINGS ---
MESH_SETTINGS = {
    'lines_per_wavelength': 15,   # tet density (steps per wavelength)
    'curvature_order': 2,         # order-2 curved elements -> slim carries curvature
    'normal_tolerance_deg': 62.5,
    'srf_mesh_gradation': 1.5,
    'vol_mesh_gradation': 1.5,
}

# --- EIGENMODE SOLVER SETTINGS ---
SOLVER_SETTINGS = {
    'number_of_modes': 1,
    'order_tet': 3,
    'accuracy': 1e-6,
    'aks_accuracy': 1e-12,
    'max_threads': 1024,
    'max_cpu_devices': 2,
}

# --- FILE PATHS & EXECUTION MODE ---
CST_PROJECT_FILE = r"C:\Users\rajab\Job\Simulations&Scripts\TESLA_nb_coating_lib\CST\Single_Tesla_Slim_1.cst"
# "new"      = build geometry from scratch.
# "existing" = attach to open project and only update solver/mesh settings.
CST_MODE = "new"


# =====================================================================
# 2. CST CONNECTION & INJECTION LOGIC (Embedded)
# =====================================================================
def connect_to_cst(cst_project_file, mode="new"):
    """Connect to CST and return the project handle."""
    try:
        logger.info("  -> importing cst.interface ...")
        import cst.interface
        logger.info("  -> attaching / launching CST design environment ...")
        de = cst.interface.DesignEnvironment.connect_to_any_or_new()
        logger.info("  -> design environment ready.")

        if mode == "existing":
            logger.info("Connecting to existing project: %s", cst_project_file)
            try:
                prj = de.get_open_project(cst_project_file)
                logger.info("  -> Project already open. Connected.")
            except Exception:
                logger.info("  -> Opening project from disk...")
                prj = de.open_project(cst_project_file)
        else:
            logger.info("Creating new MWS project...")
            prj = de.new_mws()
            logger.info("  -> new MWS project created.")

        return prj
    except ImportError as e:
        logger.error("CST Library (cst.interface) not found: %s", e)
        return None
    except Exception as e:
        logger.critical("CST Connection Failed: %s", e)
        return None


def save_project(prj, filepath):
    """Saves the CST project to an explicit path, overwriting if it already exists."""
    try:
        prj.save(filepath, allow_overwrite=True)
        logger.info("Project saved to: %s", filepath)
    except Exception as e:
        logger.error("Failed to save project: %s", e)


def inject_history(prj, block_name, vba_code):
    """Injects a named VBA block into the CST History List."""
    if not prj:
        logger.warning("Cannot inject history: Project handle is None.")
        return
    try:
        prj.model3d.add_to_history(block_name, vba_code)
        logger.info("Injected VBA block: %s", block_name)
    except Exception as e:
        logger.error("Failed to inject history block '%s': %s", block_name, e)


# =====================================================================
# 3. GEOMETRY MATH (Tangent point calculations)
# =====================================================================
def fmt(val): return f"{val:.9f}"


def validate_cup(p):
    """Fail early with a readable message instead of a CST 'curve not closed' error."""
    name = p.get('name', 'Unknown')
    if p['Beq'] + p['Bir'] >= p['L']:
        raise ValueError(
            f"{name}: Beq + Bir = {p['Beq'] + p['Bir']:.3f} >= L = {p['L']:.3f}. "
            "Equator and iris ellipses overlap along z; no tangent line exists. "
            "Reduce Beq/Bir or increase L.")
    if p['Aeq'] >= p['Req']:
        raise ValueError(f"{name}: Aeq must be smaller than Req.")
    if p['Rir'] + p['Air'] >= p['Req'] - p['Aeq']:
        raise ValueError(
            f"{name}: iris ellipse reaches past the equator ellipse radially.")


def solve_tangent_coords(p):
    L, Req, Rir = p['L'], p['Req'], p['Rir']
    Aeq, Beq, Air, Bir = p['Aeq'], p['Beq'], p['Air'], p['Bir']

    def dr_eq(z):
        inner = np.clip(1 - (z/Beq)**2, 1e-12, 1)
        return -(Aeq * z) / (Beq**2 * np.sqrt(inner))

    def dr_ir(z):
        inner = np.clip(1 - ((L-z)/Bir)**2, 1e-12, 1)
        return -(Air * (L-z)) / (Bir**2 * np.sqrt(inner))

    def r_eq(z):
        return (Req - Aeq) + Aeq * np.sqrt(np.clip(1 - (z/Beq)**2, 1e-12, 1))

    def r_ir(z): return (Rir + Air) - Air * \
        np.sqrt(np.clip(1 - ((L - z)/Bir)**2, 1e-12, 1))

    def equations(v):
        z1, z2 = v
        m1, m2 = dr_eq(z1), dr_ir(z2)
        m_line = (r_ir(z2) - r_eq(z1)) / (z2 - z1)
        return [m1 - m_line, m2 - m_line]

    sol = root(equations, [0.99*Beq, L - 0.99*Bir])
    z_eq, z_ir = sol.x
    if not sol.success or not (0.0 < z_eq < Beq) or not (L - Bir < z_ir < L) or z_ir <= z_eq:
        raise ValueError(
            f"{p.get('name', 'Unknown')}: tangent-line solve failed "
            f"(z_eq={z_eq:.3f}, z_ir={z_ir:.3f}). Parameters are likely geometrically infeasible.")
    return z_eq, r_eq(z_eq), z_ir, r_ir(z_ir)


def get_chain_points(p):
    validate_cup(p)
    z_eq_tan, r_eq_tan, z_ir_tan, r_ir_tan = solve_tangent_coords(p)

    # Iris arc: circle of radius Air, scaled in z by Bir/Air about its center
    scale_factor = p['Bir'] / p['Air']
    center_ir_r, center_ir_z = p['Rir'] + p['Air'], p['L']
    z_ir_circle_end = center_ir_z + (z_ir_tan - center_ir_z) / scale_factor

    # Equator arc: circle of radius Aeq, scaled in z by Beq/Aeq about its center.
    # p4_raw is the tangent point mapped back onto the unscaled circle.
    scale_factor_eq = p['Beq'] / p['Aeq']
    center_eq_r, center_eq_z = p['Req'] - p['Aeq'], 0.0
    z_eq_circle_start = center_eq_z + \
        (z_eq_tan - center_eq_z) / scale_factor_eq

    return {
        'p0': (0.0, 0.0), 'p1': (0.0, p['L']), 'p2': (p['Rir'], p['L']),
        'p3_raw': (r_ir_tan, z_ir_circle_end), 'p3_final': (r_ir_tan, z_ir_tan),
        'p4_raw': (r_eq_tan, z_eq_circle_start), 'p4': (r_eq_tan, z_eq_tan),
        'p5': (p['Req'], 0.0),
        'ir_center': (center_ir_r, center_ir_z),
        'eq_center': (center_eq_r, center_eq_z),
        'scale_factor': scale_factor,
        'scale_factor_eq': scale_factor_eq,
        'name': p.get('name', 'Unknown'),
    }


# =====================================================================
# 4. VBA GENERATORS
# =====================================================================
def build_simulation_setup_vba():
    s = SIMULATION_SETUP
    b, sym = s['boundaries'], s['symmetry']

    vba = f'Solver.FrequencyRange "{s["frequency_range"][0]}", "{s["frequency_range"][1]}"\n'
    vba += f'ChangeSolverType "{s["solver_type"]}"\n'

    vba += 'With Boundary\n'
    vba += f'  .Xmin "{b["xmin"]}": .Xmax "{b["xmax"]}"\n'
    vba += f'  .Ymin "{b["ymin"]}": .Ymax "{b["ymax"]}"\n'
    vba += f'  .Zmin "{b["zmin"]}": .Zmax "{b["zmax"]}"\n'
    vba += f'  .Xsymmetry "{sym["x"]}": .Ysymmetry "{sym["y"]}": .Zsymmetry "{sym["z"]}"\n'
    vba += 'End With\n'

    # Background: zero space around the structure
    vba += 'With Background\n  .ResetBackground\n'
    vba += '  .XminSpace "0.0": .XmaxSpace "0.0": .YminSpace "0.0"\n'
    vba += '  .YmaxSpace "0.0": .ZminSpace "0.0": .ZmaxSpace "0.0"\n'
    vba += '  .ApplyInAllDirections "False"\nEnd With\n'
    # Background material kept as PEC (user decision)
    vba += f'With Material\n  .Reset\n  .Type "{s["background_material"]}"\n  .ChangeBackgroundMaterial\nEnd With\n'
    return vba


def build_mesh_settings_vba():
    m = MESH_SETTINGS
    lpw = m['lines_per_wavelength']

    # Density block (steps per wavelength / per box)
    vba = 'With MeshSettings\n'
    vba += '  .SetMeshType "Tet"\n'
    vba += f'  .Set "StepsPerWaveNear", "{lpw}"\n  .Set "StepsPerWaveFar", "{lpw}"\n'
    vba += f'  .Set "StepsPerBoxNear", "{lpw}"\n  .Set "StepsPerBoxFar", "{lpw}"\n'
    vba += 'End With\n'

    # Recorded HF mesh blocks, verbatim (order-2 curvature etc.)
    vba += f'''With MeshSettings 
     .SetMeshType "Tet" 
     .Set "CurvatureOrder", "{m['curvature_order']}" 
     .Set "CurvatureOrderPolicy", "fixedorder" 
     .Set "CurvRefinementControl", "NormalTolerance" 
     .Set "NormalTolerance", "{m['normal_tolerance_deg']}" 
     .Set "SrfMeshGradation", "{m['srf_mesh_gradation']}" 
     .Set "SrfMeshOptimization", "0" 
        End With 
        With MeshSettings 
            .SetMeshType "Unstr" 
            .Set "UseMaterials",  "1" 
            .Set "MoveMesh", "0" 
        End With 
        With MeshSettings 
            .SetMeshType "Tet" 
            .Set "UseAnisoCurveRefinement", "0" 
            .Set "UseSameSrfAndVolMeshGradation", "1" 
            .Set "VolMeshGradation", "{m['vol_mesh_gradation']}" 
            .Set "VolMeshOptimization", "0" 
        End With 
        With MeshSettings 
            .SetMeshType "Unstr" 
            .Set "UseIgnoreFeatures", "0" 
            .Set "FeatureTolerance", "1e-05" 
            .Set "AutomaticFeatureTolerance", "1" 
            .Set "UseCoincidenceTolerance", "0" 
            .Set "CoincidenceTolerance", "1e-05" 
            .Set "AutomaticCoincidenceTolerance", "1" 
            .Set "SelfIntersectionCheck", "1" 
        End With 
        With Mesh 
            .SetParallelMesherMode "Tet", "maximum" 
            .SetMaxParallelMesherThreads "Tet", "1" 
        End With 
        '''
    return vba


def build_solver_settings_vba():
    s = SOLVER_SETTINGS

    vba = 'Mesh.SetFlavor "High Frequency"\n'
    vba += 'Mesh.SetCreator "High Frequency"\n'
    vba += 'EigenmodeSolver.Reset\n'

    vba += 'With Solver\n'
    vba += '     .CalculationType "Eigenmode"\n'
    vba += '     .AKSReset\n'
    vba += '     .AKSPenaltyFactor "1"\n'
    vba += '     .AKSEstimation "0"\n'
    vba += '     .AKSAutomaticEstimation "True"\n'
    vba += '     .AKSEstimationCycles "5"\n'
    vba += '     .AKSIterations "2"\n'
    vba += f'     .AKSAccuracy "{s["aks_accuracy"]}"\n'
    vba += 'End With\n'

    vba += 'With EigenmodeSolver\n'
    vba += '     .SetMethodType "AKS", "Hex"\n'
    vba += '     .SetMethodType "Automatic", "Tet"\n'
    vba += '     .SetMeshType "Tetrahedral Mesh"\n'
    vba += '     .SetMeshAdaptationHex "False"\n'
    vba += '     .SetMeshAdaptationTet "False"\n'
    vba += f'     .SetNumberOfModes "{s["number_of_modes"]}"\n'
    vba += '     .SetStoreResultsInCache "False"\n'
    vba += '     .SetCalculateExternalQFactor "False"\n'
    vba += '     .SetConsiderStaticModes "True"\n'
    vba += '     .SetCalculateThermalLosses "True"\n'
    vba += '     .SetModesInFrequencyRange "False"\n'
    vba += '     .SetFrequencyTarget "True", "0.0"\n'
    vba += f'     .SetAccuracy "{s["accuracy"]}"\n'
    vba += '     .SetQExternalAccuracy "1e-4"\n'
    vba += '     .SetMaterialEvaluationFrequency "True", ""\n'
    vba += '     .SetTDCompatibleMaterials "False"\n'
    vba += f'     .SetOrderTet "{s["order_tet"]}"\n'
    vba += '     .SetUseSensitivityAnalysis "False"\n'
    vba += '     .SetConsiderLossesInPostprocessingOnly "True"\n'
    vba += '     .SetMinimumQ "1.0"\n'
    vba += '     .SetUseParallelization "True"\n'
    vba += f'     .SetMaxNumberOfThreads "{s["max_threads"]}"\n'
    vba += f'     .MaximumNumberOfCPUDevices "{s["max_cpu_devices"]}"\n'
    vba += '     .SetRemoteCalculation "False"\n'
    vba += 'End With\n'

    vba += 'UseDistributedComputingForParameters "False"\n'
    vba += 'MaxNumberOfDistributedComputingParameters "2"\n'
    vba += 'UseDistributedComputingMemorySetting "False"\n'
    vba += 'MinDistributedComputingMemoryLimit "0"\n'
    vba += 'UseDistributedComputingSharedDirectory "False"\n'
    vba += 'OnlyConsider0D1DResultsForDC "False"\n'
    return vba


def build_9cell_vacuum_vba():
    _cup_map = {'END_CUP_1': END_CUP_1,
                'END_CUP_2': END_CUP_2, 'MID_CUP': MID_CUP}
    left_cup, right_cup = _cup_map[CAVITY['end_left']
                                   ], _cup_map[CAVITY['end_right']]

    layout = [(left_cup, True)]
    for _ in range(CAVITY['num_cells'] - 1):
        layout.extend([(MID_CUP, False), (MID_CUP, True)])
    layout.append((right_cup, False))

    vba = 'Component.New "TESLA"\n'
    proto_map = {id(END_CUP_1): "Proto_End1", id(MID_CUP)
                    : "Proto_Mid", id(END_CUP_2): "Proto_End2"}
    seen, prototypes = {}, []

    for cup, _ in layout:
        if id(cup) not in seen:
            seen[id(cup)] = True
            prototypes.append((cup, proto_map[id(cup)]))

    # Draw and revolve prototypes
    for p_def, proto_name in prototypes:
        pts = get_chain_points(p_def)
        curve_name = f"C_{proto_name}"
        vba += f'Curve.NewCurve "{curve_name}"\n'
        vba += f'With Line\n.Reset\n.Name "S1"\n.Curve "{curve_name}"\n.X1 "{fmt(pts["p0"][0])}"\n.Y1 "{fmt(pts["p0"][1])}"\n.X2 "{fmt(pts["p1"][0])}"\n.Y2 "{fmt(pts["p1"][1])}"\n.Create\nEnd With\n'
        vba += f'With Line\n.Reset\n.Name "S2"\n.Curve "{curve_name}"\n.X1 "{fmt(pts["p1"][0])}"\n.Y1 "{fmt(pts["p1"][1])}"\n.X2 "{fmt(pts["p2"][0])}"\n.Y2 "{fmt(pts["p2"][1])}"\n.Create\nEnd With\n'

        # S3: iris arc (circle of radius Air, then scaled in z by Bir/Air)
        vba += f'With Arc\n.Reset\n.Name "S3"\n.Curve "{curve_name}"\n.Orientation "CounterClockwise"\n.Xcenter "{fmt(pts["ir_center"][0])}"\n.Ycenter "{fmt(pts["ir_center"][1])}"\n.X1 "{fmt(pts["p2"][0])}"\n.Y1 "{fmt(pts["p2"][1])}"\n.X2 "{fmt(pts["p3_raw"][0])}"\n.Y2 "{fmt(pts["p3_raw"][1])}"\n.UseAngle "False"\n.Create\nEnd With\n'
        vba += f'SelectTreeItem "Curves\\{curve_name}\\S3"\nWith Transform\n.Reset\n.Name "{curve_name}:S3"\n.Origin "Free"\n.Center "{fmt(pts["ir_center"][0])}", "{fmt(pts["ir_center"][1])}", "0"\n.ScaleFactor "1", "{fmt(pts["scale_factor"])}", "1"\n.Transform "Curve", "Scale"\nEnd With\n'

        vba += f'With Line\n.Reset\n.Name "S4"\n.Curve "{curve_name}"\n.X1 "{fmt(pts["p3_final"][0])}"\n.Y1 "{fmt(pts["p3_final"][1])}"\n.X2 "{fmt(pts["p4"][0])}"\n.Y2 "{fmt(pts["p4"][1])}"\n.Create\nEnd With\n'

        # S5: equator arc (circle of radius Aeq, then scaled in z by Beq/Aeq).
        # Starts at p4_raw (tangent point on the UNSCALED circle); the scale
        # transform moves it onto p4 so S4 and S5 meet and the profile closes.
        vba += f'With Arc\n.Reset\n.Name "S5"\n.Curve "{curve_name}"\n.Orientation "Clockwise"\n.Xcenter "{fmt(pts["eq_center"][0])}"\n.Ycenter "{fmt(pts["eq_center"][1])}"\n.X1 "{fmt(pts["p4_raw"][0])}"\n.Y1 "{fmt(pts["p4_raw"][1])}"\n.X2 "{fmt(pts["p5"][0])}"\n.Y2 "{fmt(pts["p5"][1])}"\n.UseAngle "False"\n.Create\nEnd With\n'
        if abs(pts["scale_factor_eq"] - 1.0) > 1e-12:
            vba += f'SelectTreeItem "Curves\\{curve_name}\\S5"\nWith Transform\n.Reset\n.Name "{curve_name}:S5"\n.Origin "Free"\n.Center "{fmt(pts["eq_center"][0])}", "{fmt(pts["eq_center"][1])}", "0"\n.ScaleFactor "1", "{fmt(pts["scale_factor_eq"])}", "1"\n.Transform "Curve", "Scale"\nEnd With\n'

        vba += f'With Line\n.Reset\n.Name "S6"\n.Curve "{curve_name}"\n.X1 "{fmt(pts["p5"][0])}"\n.Y1 "{fmt(pts["p5"][1])}"\n.X2 "{fmt(pts["p0"][0])}"\n.Y2 "{fmt(pts["p0"][1])}"\n.Create\nEnd With\n'

        vba += f'With CoverCurve\n.Reset\n.Name "Face_{proto_name}"\n.Component "TESLA"\n.Material "Vacuum"\n.Curve "{curve_name}"\n.Create\nEnd With\n'
        vba += f'Pick.PickFaceFromId "TESLA:Face_{proto_name}", "1"\nPick.AddEdge "0","0","0","0","1","0"\n'
        vba += f'With Rotate\n.Reset\n.Name "{proto_name}"\n.Component "TESLA"\n.Material "Vacuum"\n.Mode "Picks"\n.Angle "360"\n.NSteps "0"\n.Create\nEnd With\n'
        vba += f'Solid.Delete "TESLA:Face_{proto_name}"\nCurve.DeleteCurve "{curve_name}"\n'

    # Assemble, Translate, and Merge
    current_z = 0.0
    for i, (p, is_flipped) in enumerate(layout):
        solid_name, source, L = f"Solid_{i}", proto_map[id(p)], p['L']
        vba += f'With Transform\n.Reset\n.Name "TESLA:{source}"\n.Vector "0", "0", "0"\n.MultipleObjects "True"\n.Transform "Shape", "Translate"\nEnd With\n'
        vba += f'Solid.Rename "TESLA:{source}_1", "{solid_name}"\n'
        if is_flipped:
            vba += f'With Transform\n.Reset\n.Name "TESLA:{solid_name}"\n.Origin "Free"\n.Center "0", "0", "0"\n.ScaleFactor "1", "-1", "1"\n.Transform "Shape", "Scale"\nEnd With\n'
            move_z = L + current_z
        else:
            move_z = current_z
        vba += f'With Transform\n.Reset\n.Name "TESLA:{solid_name}"\n.Vector "0", "{fmt(move_z)}", "0"\n.Transform "Shape", "Translate"\nEnd With\n'
        current_z += L

    total_length = current_z
    for _, proto_name in prototypes:
        vba += f'Solid.Delete "TESLA:{proto_name}"\n'

    vba += f'Solid.Rename "TESLA:Solid_0", "Vacuum_Volume0"\n'
    for i in range(1, len(layout)):
        vba += f'Solid.Add "TESLA:Vacuum_Volume0", "TESLA:Solid_{i}"\n'

    # Center the cavity
    shift_val = -total_length / 2.0
    vba += f'With Transform\n.Reset\n.Name "TESLA:Vacuum_Volume0"\n.Origin "Free"\n.Center "0","0","0"\n.Angle "90","0","0"\n.Transform "Shape", "Rotate"\nEnd With\n'
    vba += f'With Transform\n.Reset\n.Name "TESLA:Vacuum_Volume0"\n.Vector "0", "0", "{fmt(shift_val)}"\n.Transform "Shape", "Translate"\nEnd With\n'

    # Add Beam Pipes
    P_LEN, P_RAD = PIPE_CONFIG['length'], PIPE_CONFIG['radius']
    z_left, z_right = total_length / 2.0 + P_LEN, -total_length / 2.0

    vba += f'Curve.NewCurve "PipeCurve"\nWith Circle\n.Reset\n.Name "PC"\n.Curve "PipeCurve"\n.Radius "{P_RAD}"\n.Xcenter "0"\n.Ycenter "0"\n.Segments "0"\n.Create\nEnd With\n'
    vba += f'With CoverCurve\n.Reset\n.Name "Pipe_R_In"\n.Component "TESLA"\n.Material "Vacuum"\n.Curve "PipeCurve:PC"\n.DeleteCurve "True"\n.Create\nEnd With\n'
    vba += f'Solid.ThickenSheetAdvanced "TESLA:Pipe_R_In","Inside","{P_LEN}","True"\n'

    vba += f'With Transform\n.Reset\n.Name "TESLA:Pipe_R_In"\n.Vector "0","0","0"\n.MultipleObjects "True"\n.Transform "Shape", "Translate"\nEnd With\n'
    vba += f'Solid.Rename "TESLA:Pipe_R_In_1", "Pipe_L_In"\n'
    vba += f'With Transform\n.Reset\n.Name "TESLA:Pipe_L_In"\n.Vector "0","0","{fmt(z_left)}"\n.Transform "Shape", "Translate"\nEnd With\n'
    vba += f'With Transform\n.Reset\n.Name "TESLA:Pipe_R_In"\n.Vector "0","0","{fmt(z_right)}"\n.Transform "Shape", "Translate"\nEnd With\n'

    vba += f'Solid.Add "TESLA:Vacuum_Volume0","TESLA:Pipe_R_In"\nSolid.Add "TESLA:Vacuum_Volume0","TESLA:Pipe_L_In"\n'
    return vba


# =====================================================================
# 5. EXECUTION LOGIC
# =====================================================================
def run_cavity_setup():
    """Connects to CST, builds the cavity, sets up mesh + solver, and saves."""
    logger.info("Connecting to CST Studio Suite...")

    prj = connect_to_cst(CST_PROJECT_FILE, mode=CST_MODE)
    if not prj:
        logger.error("Failed to connect to CST. Is it running?")
        return

    if CST_MODE == "new":
        logger.info("Injecting 9-cell TESLA geometry VBA into history list...")
        geo_vba = build_9cell_vacuum_vba()
        inject_history(prj, "Build 9-Cell Geometry", geo_vba)
    else:
        logger.info("Attaching to existing project (skipping geometry build).")

    logger.info(
        "Applying simulation setup (Boundaries, Frequency, PEC background)...")
    sim_vba = build_simulation_setup_vba()
    inject_history(prj, "Simulation Setup", sim_vba)

    logger.info("Applying HF mesh settings (curvature order %d, %d lines/wavelength)...",
                MESH_SETTINGS['curvature_order'], MESH_SETTINGS['lines_per_wavelength'])
    mesh_vba = build_mesh_settings_vba()
    inject_history(prj, "Mesh Settings", mesh_vba)

    logger.info("Applying eigenmode solver settings (AKS, Tet order %d, %d mode(s))...",
                SOLVER_SETTINGS['order_tet'], SOLVER_SETTINGS['number_of_modes'])
    solver_vba = build_solver_settings_vba()
    inject_history(prj, "Eigenmode Solver Settings", solver_vba)

    if CST_MODE == "new":
        project_path = Path(CST_PROJECT_FILE)
        project_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving project to: %s", project_path)
        save_project(prj, str(project_path))

    logger.info("Cavity setup complete.")
