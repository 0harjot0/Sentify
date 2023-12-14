from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    source_url: str 
    local_data_file: Path 
    data_path: Path

@dataclass(frozen=True)
class DataValidationConfig:
    pre_process_file: Path 
    post_process_files: list 
    data_path: Path

@dataclass(frozen=True)
class DataPreparationConfig:
    data_path: Path 
    pre_process_file: Path 
    col_names: list 
    text_col: str 
    target_col: str 