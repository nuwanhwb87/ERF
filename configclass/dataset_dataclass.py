import json
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from typing import List

@dataclass
class DatasetConfig:
    folders: List[str]
    long_names: List[str]
    short_names: List[str]
    field_folders: List[str]
    field_shorts: List[str]
    height: int
    width: int
    chunk_size: int
    num_variables: int
    num_static_fields: int
    max_horizon: int # Maximum time horizon for the model. Used for scaling time embedding and making sure we don't go outside dataset
    file_directory: str
    save_directory: str
    # dataset time config
    start_datetime: str
    end_datetime: str
    time_freq: str  # e.g., "1H", "6H", "7D", "W", "W-TUE"
    # dataset split config
    split_mode: str  # "years" | "dates"
    train_year_end: Optional[int]
    val_year_end: Optional[int]
    train_until: Optional[str]
    val_until: Optional[str]
    #skip
    skip: str
    land_only: bool = True
    lsm_path: Optional[str] = None

    @staticmethod
    def from_json(path: Path) -> "DatasetConfig":
        with path.open("r") as f:
            d = json.load(f)
        return DatasetConfig(**d)