from sentify.constants import *
from utils.common import read_yaml, create_directories
from sentify.entity import (
    DataIngestionConfig
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
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        '''
        creates and returns data ingestion configuration
        
        ## Parameters: 
        
        None 
        
        ## Returns:
        
        data_ingestion_config: DataIngestionConfig
            the configuration for data ingestion 
        '''
        config = self.config.data_ingestion 
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir, 
            source_url=config.source_url, 
            local_data_file=config.local_data_file
        )
        
        return data_ingestion_config
