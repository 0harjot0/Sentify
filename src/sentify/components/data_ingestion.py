import os 
import gdown
from zipfile import ZipFile
from pathlib import Path 

from logger import logger 
from utils.common import get_size, create_directories
from sentify.config.configuration import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        '''
        creates an instance for Data Ingestion class for all the data ingestion techniques
        
        ## Parameters: 
        
        config: DataIngestionConfig
            configuration for data ingestion 
        '''
        self.config = config 
        
    def download_data(self):
        '''
        downloads the data, if not already available at the configured location
        '''
        create_directories([self.config.data_path])
        if not os.path.exists(self.config.local_data_file):
            filename = gdown.download(self.config.source_url, 
                                      self.config.local_data_file)
            logger.info(f"{filename} downloaded! Saved at - {self.config.local_data_file}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_data(self):
        '''
        extracts the data zip file
        '''
        create_directories([self.config.data_path])
        with ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(self.config.data_path)
            
        logger.info(f"Extracted data in {self.config.data_path} path")
        