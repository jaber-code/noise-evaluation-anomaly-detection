import yaml
from pydantic import BaseModel, field_validator, model_validator
from typing import Dict, Optional

from enums import AggregationType, TrainingType, ModelType, NoiseType, NoiseGenerationMode


class ExperimentConfig(BaseModel):
    dataset_name: str
    agg_type: AggregationType
    seed: int
    write_to_file: bool = False
    output_dir: str = "runs"


class TrainingConfig(BaseModel):
    training_type: TrainingType
    model_type: ModelType
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    lr_decay_epoch: int
    lr_decay_factor: float

    @field_validator("epochs", "batch_size")
    @classmethod
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError(f"must be positive, got {v}")
        return v

    @field_validator("lr", "weight_decay", "lr_decay_factor")
    @classmethod
    def must_be_positive_float(cls, v):
        if v <= 0:
            raise ValueError(f"must be positive, got {v}")
        return v

    @field_validator("lr_decay_epoch")
    @classmethod
    def decay_epoch_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError(f"lr_decay_epoch must be positive, got {v}")
        return v


# Noise types that don't use sigma_max
_SIGMA_FREE_TYPES = {NoiseType.SALT_PEPPER, NoiseType.BERNOULLI}


class NoiseConfig(BaseModel):
    sigma_max: Optional[float] = None
    noise_ratios: list[float]

    @field_validator("noise_ratios")
    @classmethod
    def ratios_in_unit_interval(cls, v):
        if len(v) == 0:
            raise ValueError("noise_ratios must contain at least one value")
        for r in v:
            if not (0.0 < r <= 1.0):
                raise ValueError(f"each noise_ratio must be in (0, 1], got {r}")
        return v

    @field_validator("sigma_max")
    @classmethod
    def sigma_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError(f"sigma_max must be positive, got {v}")
        return v


class NoiseDiversityConfig(BaseModel):
    #noise_generation_mode: NoiseGenerationMode
    #number_of_noise_samples_per_clean_sample: int
    m: int
    noise_dist_configs: Dict[NoiseType, NoiseConfig]
    used_noise_distributions: list[NoiseType]

    """@field_validator("m", "number_of_noise_samples_per_clean_sample")
    @classmethod
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError(f"must be positive, got {v}")
        return v"""

    @model_validator(mode="after")
    def used_distributions_must_be_configured(self):
        for noise_type in self.used_noise_distributions:
            if noise_type not in self.noise_dist_configs:
                raise ValueError(
                    f"{noise_type} is in used_noise_distributions but has no entry in noise_dist_configs"
                )
        return self


class Config(BaseModel):
    experiment: ExperimentConfig
    training: TrainingConfig
    noise_diversity: NoiseDiversityConfig

    def write_summary(self, write_line) -> None:
        """write_line: callable taking a string (e.g. DualLogger.write or print)."""
        write_line("=" * 60)
        write_line("CONFIGURATION")
        write_line("=" * 60)

        write_line("\n[experiment]")
        for k, v in self.experiment.model_dump().items():
            write_line(f"  {k}: {v}")

        write_line("\n[training]")
        for k, v in self.training.model_dump().items():
            write_line(f"  {k}: {v}")

        write_line("\n[noise_diversity]")
        nd = self.noise_diversity
        #write_line(f"  noise_generation_mode: {nd.noise_generation_mode.value}")
        #write_line(f"  number_of_noise_samples_per_clean_sample: {nd.number_of_noise_samples_per_clean_sample}")
        write_line(f"  m: {nd.m}")
        write_line(f"  used_noise_distributions: {[nt.value for nt in nd.used_noise_distributions]}")
        write_line("  noise_dist_configs (only used):")
        for nt in nd.used_noise_distributions:
            cfg = nd.noise_dist_configs[nt]
            sigma = cfg.sigma_max if cfg.sigma_max is not None else "—"
            write_line(f"    {nt.value}: sigma_max={sigma}, noise_ratio={cfg.noise_ratios}")

        write_line("=" * 60)


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r") as f:
        return Config(**yaml.safe_load(f))