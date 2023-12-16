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
    bert_tokenizer: Path 
    bert_model: Path 
    labels: Path
    
@dataclass(frozen=True)
class ModelTrainerConfig:
    activation: str 
    optimizer: str 
    metrics: list 
    vector_size: int 
    fasttext_seq_len: int 
    bert_seq_len: int 
    fasttext_embedding: Path 
    fasttext_embed_matrix: Path 
    bert_embedding: Path 
    labels: Path 
    fasttext_model: Path 
    fasttext_tokenizer: Path 
    epochs: int
    models_path: Path

@dataclass(frozen=True)
class TweetScraperConfig:
    log_level: int 
    skip_instance_check: int 
    
@dataclass(frozen=True)
class ModelPredictionConfig:
    active_model: str 
    models_path: Path 
    model_name: str 
    fasttext_model: Path 
    fasttext_tokenizer: Path 
    bert_tokenizer: Path 
    bert_model: Path 
    fasttext_seq_len: int
    bert_seq_len: int 
