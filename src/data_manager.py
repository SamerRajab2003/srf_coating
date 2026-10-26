"""
Handles persistent storage of run configurations, raw extracted fields, and statistical results.
"""
import yaml
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)


def create_run_directory(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(config_dict: dict, run_dir: Path):
    snapshot_path = run_dir / "config_snapshot.yaml"

    def stringify_paths(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: stringify_paths(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [stringify_paths(i) for i in obj]
        return obj

    with open(snapshot_path, 'w') as f:
        yaml.dump(stringify_paths(config_dict), f,
                  default_flow_style=False, sort_keys=False)


def save_extracted_fields(
    run_dir: Path, vertices_mm: np.ndarray, triangle_node_indices: np.ndarray,
    Jx_nodes: np.ndarray, Jy_nodes: np.ndarray, Jz_nodes: np.ndarray,
    frequency_hz: float, stored_energy_J: float
):
    np.savez_compressed(
        run_dir / "extracted_fields.npz",
        vertices_mm=vertices_mm, triangle_node_indices=triangle_node_indices,
        Jx_nodes=Jx_nodes, Jy_nodes=Jy_nodes, Jz_nodes=Jz_nodes,
        frequency_hz=frequency_hz, stored_energy_J=stored_energy_J
    )


def save_raw_samples(
    results_db: Dict[float, dict],
    length_scales_mm: List[float],
    Q_uniform: float,
    run_dir: Path
):
    sample_rows = []
    for l_mm in length_scales_mm:
        q_norm = results_db[l_mm]["Q_normalized"]
        q_vals = q_norm * Q_uniform
        df_vals = results_db[l_mm]["delta_f_hz"]
        fhot_vals = results_db[l_mm]["f_hot"]

        for idx in range(len(q_norm)):
            sample_rows.append({
                "length_scale_mm": float(l_mm),
                "sample_index": idx,
                "Q": q_vals[idx],
                "Q_normalized": q_norm[idx],
                "delta_f_hz": df_vals[idx],
                "f_hot": fhot_vals[idx],
            })
    pd.DataFrame(sample_rows).to_csv(
        run_dir / "all_samples_raw.csv", index=False)
    logger.info("Saved raw samples to %s", run_dir / "all_samples_raw.csv")
