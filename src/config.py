"""
Pydantic schema for the YAML configuration file.
"""
import logging
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, ValidationError
from src.errors import ConfigError

logger = logging.getLogger(__name__)


class ProjectConfig(BaseModel):
    cst_project_file: Path
    slim_file: Path | None = None
    mode_number: int = Field(default=1, ge=1)


class SurfaceConfig(BaseModel):
    metric: str = Field(default="auto", pattern="^(auto|flat|curved)$")
    cst_q_post_processing: bool = False


class CoatingConfig(BaseModel):
    mean_thickness_nm: float = Field(default=260.0, gt=0.0)
    max_deviation_nm: float = Field(default=26.0, gt=0.0)


class SisParameters(BaseModel):
    T_op_K: float = 4.2
    sigma_1_S_per_m: float = 2.86e6
    lambda_1_0_m: float = 180.57e-9
    Tc_1_K: float = 15.4
    d_I_m: float = 10.0e-9
    sigma_2_300K_S_per_m: float = 6.58e6
    RRR_2: float = 300.0
    lambda_2_0_m: float = 39.0e-9
    Tc_2_K: float = 9.23


class LayerConfig(BaseModel):
    name: str
    type: str = Field(pattern="^(superconductor|normal|insulator)$")
    thickness_nm: float | None = None
    sigma_S_per_m: float | None = None
    sigma_300K_S_per_m: float | None = None
    RRR: float | None = None
    lambda_0_m: float | None = None
    Tc_K: float | None = None
    epsilon_r: float = 1.0
    tan_delta: float = 0.0


class ImpedanceConfig(BaseModel):
    model_type: str = "transfer_matrix"
    T_op_K: float = 4.2
    parameters: SisParameters = SisParameters()
    layers: list[LayerConfig] = []

    def resolved_layers(self) -> list[LayerConfig]:
        if self.layers:
            return self.layers
        p = self.parameters
        return [
            LayerConfig(name="coating", type="superconductor",
                        sigma_S_per_m=p.sigma_1_S_per_m,
                        lambda_0_m=p.lambda_1_0_m, Tc_K=p.Tc_1_K),
            LayerConfig(name="insulator", type="insulator",
                        thickness_nm=p.d_I_m * 1e9,
                        epsilon_r=10.4, tan_delta=2.4e-4),
            LayerConfig(name="substrate", type="superconductor",
                        sigma_300K_S_per_m=p.sigma_2_300K_S_per_m, RRR=p.RRR_2,
                        lambda_0_m=p.lambda_2_0_m, Tc_K=p.Tc_2_K),
        ]


class StatisticsConfig(BaseModel):
    """Stochastic Monte Carlo and SPDE mesh correlation parameters."""
    num_samples: int = Field(default=2048, ge=1)
    length_scales_mm: list[float] = [3.0, 6.0, 12.0,
                                     25.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    base_seed: int = 123456


class OutputConfig(BaseModel):
    directory: Path = Path("outputs")


class RunConfig(BaseModel):
    project: ProjectConfig
    surface: SurfaceConfig = SurfaceConfig()
    coating: CoatingConfig = CoatingConfig()
    impedance: ImpedanceConfig = ImpedanceConfig()
    statistics: StatisticsConfig = StatisticsConfig()
    output: OutputConfig = OutputConfig()


def derive_slim_path(cst_project_file: Path) -> Path:
    return cst_project_file.parent / cst_project_file.stem / "Result" / "3d.slim"


def load_config(config_path: Path) -> RunConfig:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Config file {config_path} is not valid YAML: {e}") from e

    try:
        cfg = RunConfig(**raw)
    except ValidationError as e:
        details = "\n".join(
            f"  - {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors())
        raise ConfigError(
            f"Invalid values in {config_path}:\n{details}") from e

    if cfg.project.slim_file is None:
        cfg.project.slim_file = derive_slim_path(cfg.project.cst_project_file)
        logger.debug("slim_file not set; derived %s", cfg.project.slim_file)
    logger.info("Config loaded: %s", config_path)
    return cfg
