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
    
@dataclass(frozen=True)
class DataTransformationConfig:
    data_path: Path 
    bert_seq_len: int 
    fasttext_seq_len: int 
    vector_size: int 
    model_name: str 
    fasttext_epochs: int 
    word_count: int
    fasttext_embedding: Path 
    fasttext_embed_matrix: Path 
    fasttext_model: Path 
    fasttext_tokenizer: Path
    bert_embedding: Path
    labels: Path
    