from sentify.config.configuration import DataValidationConfig
from utils.common import get_size
from logger import logger

import os 


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        '''
        creates an instance for Data Validation class, it validates the 
        structure of the input data for the model
        
        ## Parameters: 
        
        config: DataValidationConfig
            configuration for the data validation 
        '''
        self.config = config 
        
    def pre_process_validation(self):
        '''
        checks for the validation of the input data structure and ordering for the model 
        '''  
        if os.path.exists(self.config.pre_process_file):
            logger.info(f"Data file validated for the processing at 
                        {self.config.pre_process_file} of size {get_size(self.config.pre_process_file)}")
            
        else: 
            raise Exception(f"Data file wasn't validated for processing at 
                            {self.config.pre_process_file}")
            
    def post_process_validation(self):
        '''
        checks for the validation of the processed data structure and ordering for the model
        '''
        all_files = os.listdir(self.config.data_path)
        
        for file in self.config.post_process_files:
            if file not in all_files:
                raise Exception(f"File {file} not found in location {self.config.data_path}")
            
        logger.info(f"Data files validated for the post processing at 
                    {self.config.data_path}")
        