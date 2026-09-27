from dataclasses import dataclass


@dataclass
class TestConfig:
    name: str # method name
    model:str
    batch_size: int
    spacing: int
    t_min: int
    t_max: int
    t_iter: int
    t_direct: int
    n_ens: int
    alpha: int
    data_directory: str
    model_directory: str
    result_directory: str
    test_start: str
    test_end: str
    land_only: bool = True
    lsm_path: str | None = None