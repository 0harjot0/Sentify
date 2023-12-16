from sentify.constants import *
from utils.common import read_yaml, create_directories
from sentify.entity import (
    DataIngestionConfig,
    DataValidationConfig, 
    DataPreparationConfig, 
    DataTransformationConfig, 
    ModelTrainerConfig, 
    TweetScraperConfig, 
    ModelPredictionConfig
)
from pathlib import Path 


class ConfigurationManager:
    def __init__(self, 
                 config_filepath: Path = CONFIG_FILE_PATH,
                 params_filepath: Path = PARAMS_FILE_PATH):
        '''
        creates a configuration manager class which manages configuration for various stages
        of development 
        
        ## Parameters:
        
        config_filepath: Path 
            configuration yaml file path 
            
        params_filepath: Path 
            parameters yaml file path 
        '''
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        '''
        creates and returns data ingestion configuration
        
        ## Parameters: 
        
        None 
        
        ## Returns:
        
        data_ingestion_config: DataIngestionConfig
            the configuration for data ingestion 
        '''        
        data_ingestion_config = DataIngestionConfig(
            source_url=self.config.source_url, 
            local_data_file=self.config.local_data_file, 
            data_path=self.config.data_path
        )
        
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        '''
        creates and returns data validation configuration
        
        ## Parameters:
        
        None 
        
        ## Returns:
        
        data_validation_config: DataValidationConfig
            the configuration for data validation
        '''
        data_validation_config = DataValidationConfig(
            pre_process_file=self.config.pre_process_file,
            post_process_files=self.config.post_process_files, 
            data_path=self.config.data_path
        )
        
        return data_validation_config
    
    def get_data_preparation_config(self) -> DataPreparationConfig:
        '''
        creates and returns data preparation configuration 
        
        ## Parameters:
        
        None 
        
        ## Returns:
        
        data_preparation_config: DataPreparationConfig
            the configuration for data preparation
        '''
        data_preparation_config = DataPreparationConfig(
            data_path=self.config.data_path,
            pre_process_file=self.config.pre_process_file,
            col_names=self.config.col_names,
            text_col=self.config.text_col, 
            target_col=self.config.target_col
        )
        
        return data_preparation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        '''
        creates and returns data transformation configuration 
        
        ## Parameters: 
        
        None 
        
        ## Returns:
        
        data_transformation_config: DataTransformationConfig
            the configuration for data transformation 
        '''
        data_transformation_config = DataTransformationConfig(
            data_path=self.config.data_path, 
            bert_seq_len=self.params.bert_argument.max_seq_length,
            fasttext_seq_len=self.params.fasttext_argument.max_seq_length,
            vector_size=self.params.fasttext_argument.vector_size,
            model_name=self.params.bert_argument.model_name,
            fasttext_epochs=self.params.fasttext_argument.epochs, 
            word_count=self.params.fasttext_argument.word_count,
            fasttext_embedding=self.config.fasttext_embedding,
            fasttext_embed_matrix=self.config.fasttext_embed_matrix,
            fasttext_model=self.config.fasttext_model,
            fasttext_tokenizer=self.config.fasttext_tokenizer,
            bert_embedding=self.config.bert_embedding,
            bert_tokenizer=self.config.bert_tokenizer,
            bert_model=self.config.bert_model,
            labels=self.config.labels, 
            embedding_paths=self.config.embedding_paths
        )
        
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        '''
        creates and returns model trainer configuration 
        
        ## Parameters:
        
        None 
        
        ## Returns:
        
        model_trainer_config: ModelTrainerConfig
            the configuration for model trainer
        '''
        model_trainer_config = ModelTrainerConfig(
            activation=self.params.activation, 
            optimizer=self.params.optimizer, 
            metrics=self.params.metrics, 
            vector_size=self.params.fasttext_argument.vector_size, 
            fasttext_seq_len=self.params.fasttext_argument.max_seq_length, 
            bert_seq_len=self.params.bert_argument.max_seq_length, 
            fasttext_embedding=self.config.fasttext_embedding, 
            fasttext_embed_matrix=self.config.fasttext_embed_matrix, 
            bert_embedding=self.config.bert_embedding, 
            labels=self.config.labels, 
            fasttext_model=self.config.fasttext_model, 
            fasttext_tokenizer=self.config.fasttext_tokenizer, 
            epochs=self.params.epochs, 
            models_path=self.config.models_path
        )
        
        return model_trainer_config
        
    def get_tweet_scraper_config(self) -> TweetScraperConfig:
        '''
        creates and returns tweet scraper configuration 
        
        ## Parameters:
        
        None 
        
        ## Returns:
        
        tweet_scraper_config: TweetScraperConfig
            the configuration for tweet scraping
        '''
        tweet_scraper_config = TweetScraperConfig(
            log_level=self.params.log_level,
            skip_instance_check=self.params.skip_instance_check
        )
        
        return tweet_scraper_config
    
    def get_model_prediction_config(self) -> ModelPredictionConfig:
        '''
        creates and returns model prediction configuration 
        
        ## Parameters:
        
        None 
        
        ## Returns:
        
        model_prediction_config: ModelPredictionConfig
            the configuration for model prediction 
        '''
        model_prediction_config = ModelPredictionConfig(
            active_model=self.params.active_model, 
            models_path=self.config.models_path, 
            model_name=self.params.bert_argument.model_name, 
            fasttext_model=self.config.fasttext_model, 
            fasttext_tokenizer=self.config.fasttext_tokenizer, 
            bert_tokenizer=self.config.bert_tokenizer, 
            bert_model=self.config.bert_model,
            fasttext_seq_len=self.params.fasttext_argument.max_seq_length,
            bert_seq_len=self.params.bert_argument.max_seq_length
        )
        
        return model_prediction_config
    