import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

DETERMINISTIC_METHOD_ALIASES = {
    "encoder_decoder_fno": {
        "use_baseline_fno": True,
        "use_fno_bottleneck": False,
        "use_fno_internal_residual": True,
        "use_fno_external_residual": True,
        "fno_modes1": -1,
        "fno_modes2": 24,
    },
    "unet": {
        "use_baseline_fno": False,
        "use_fno_bottleneck": False,
        "use_fno_internal_residual": True,
        "use_fno_external_residual": True,
        "fno_modes1": -1,
        "fno_modes2": 24,
    },
    "unet_1d_fno": {
        "use_baseline_fno": False,
        "use_fno_bottleneck": True,
        "use_fno_internal_residual": False,
        "use_fno_external_residual": False,
        "fno_modes1": -1,
        "fno_modes2": 24,
    },
    "unet_1d_fno_residual": {
        "use_baseline_fno": False,
        "use_fno_bottleneck": True,
        "use_fno_internal_residual": True,
        "use_fno_external_residual": True,
        "fno_modes1": -1,
        "fno_modes2": 24,
    },
    "unet_2d_fno": {
        "use_baseline_fno": False,
        "use_fno_bottleneck": True,
        "use_fno_internal_residual": False,
        "use_fno_external_residual": False,
        "fno_modes1": 24,
        "fno_modes2": 24,
    },
}
DETERMINISTIC_METHOD_CHOICES = tuple(DETERMINISTIC_METHOD_ALIASES)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", "none", ""}:
            return False
    return bool(value)


@dataclass
class TrainConfig:
    problem_name: str
    name: str # method name
    batch_size: int
    filters: int
    weight_decay: float
    lr: float
    epochs: int
    spacing: int
    t_min: int
    delta_t: int
    t_max: int
    conditioning_times: List[int]
    model:str
    seed: int
    variable_names: List[str]
    num_variables: int
    num_static_fields: int
    height: int
    width: int
    data_directory: str
    result_directory: str
    ckpt_dir: str
    save_every: int # save every N epochs
    max_horizon: int # Maximum time horizon for the model. Used for scaling time embedding and making sure we don't go outside dataset
    # dataset time config
    start_datetime: str
    end_datetime: str
    time_freq: str  # e.g., "1H", "6H", "7D", "W", "W-TUE"
    # dataset split config
    split_mode: str  # "years" | "dates"
    train_year_end: int | None
    val_year_end: int | None
    train_until: str | None
    val_until: str | None
    skip: str
    use_fno_bottleneck: bool = False
    resume_from_checkpoint: str | None = None
    use_baseline_fno: bool = False
    method: str | None = None
    use_fno_internal_residual: bool = True
    use_fno_external_residual: bool = True
    fno_modes1: int = -1
    fno_modes2: int = 24
    land_only: bool = True
    lsm_path: str | None = None

    def __post_init__(self):
        self.use_fno_bottleneck = _as_bool(self.use_fno_bottleneck)
        self.use_baseline_fno = _as_bool(self.use_baseline_fno)
        self.use_fno_internal_residual = _as_bool(self.use_fno_internal_residual)
        self.use_fno_external_residual = _as_bool(self.use_fno_external_residual)
        self.land_only = _as_bool(self.land_only)
        self.fno_modes1 = int(self.fno_modes1)
        self.fno_modes2 = int(self.fno_modes2)

        if isinstance(self.method, str):
            self.method = self.method.strip().lower() or None
        if self.method in {"none", "null"}:
            self.method = None
        if self.method is None:
            return
        if "deterministic" not in self.model:
            raise ValueError("method can only be used when model contains 'deterministic'.")
        if self.method not in DETERMINISTIC_METHOD_ALIASES:
            allowed = ", ".join(DETERMINISTIC_METHOD_CHOICES)
            raise ValueError(f"Unknown method {self.method!r}. Allowed values: {allowed}.")

        for key, value in DETERMINISTIC_METHOD_ALIASES[self.method].items():
            setattr(self, key, value)

    @staticmethod
    def from_json(path: Path) -> "TrainConfig":
        with path.open("r") as f:
            d = json.load(f)
        return TrainConfig(**d)