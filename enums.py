from enum import Enum

class ModelType(str, Enum):
    VANILLA_MLP = "VANILLA_MLP"
    RES_MLP = "RES_MLP"


class NoiseGenerationMode(str, Enum):
    SINGLE_DISTRIBUTION_PER_SAMPLE = "SINGLE_DISTRIBUTION_PER_SAMPLE"
    MULTIPLE_DISTRIBUTION_PER_SAMPLE = "MULTIPLE_DISTRIBUTION_PER_SAMPLE"


class NoiseType(str, Enum):
    GAUSSIAN = "GAUSSIAN"
    LAPLACE = "LAPLACE"
    UNIFORM = "UNIFORM"
    RAYLEIGH = "RAYLEIGH"
    GAMMA = "GAMMA"
    POISSON = "POISSON"
    SALT_PEPPER = "SALT_PEPPER"
    BERNOULLI = "BERNOULLI"


class AggregationType(str, Enum):
    MAX = "MAX"
    MEAN = "MEAN"
    MIN = "MIN"
    SUM = "SUM"


class TrainingType(str, Enum):
    UAD = "UAD"
    OCC = "OCC"