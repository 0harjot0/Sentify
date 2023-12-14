from sentify.constants import *
from utils.common import read_yaml, create_directories
from sentify.entity import (
    DataIngestionConfig,
    DataValidationConfig
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
    