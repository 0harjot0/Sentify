from sentify.config.configuration import DataPreparationConfig
from logger import logger

from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np 
import os 


CODECS = ['ascii', 'big5', 'big5hkscs', 'cp037', 'cp273', 'cp424', 'cp437', 'cp500', 'cp720', 'cp737', 'cp775', 'cp850', 'cp852', 'cp855',
          'cp856', 'cp857', 'cp858', 'cp860', 'cp861', 'cp862', 'cp863', 'cp864', 'cp865', 'cp866', 'cp869', 'cp874', 'cp875', 'cp932', 'cp949',
          'cp950', 'cp1006', 'cp1026', 'cp1125', 'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254', 'cp1255', 'cp1256', 'cp1257', 'cp1258',
          'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr', 'gb2312', 'gbk', 'gb18030', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2',
          'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1', 'iso8859_2', 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6',
          'iso8859_7', 'iso8859_8', 'iso8859_9', 'iso8859_10', 'iso8859_11', 'iso8859_13', 'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab', 'koi8_r', 'koi8_t',
          'koi8_u', 'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2', 'mac_roman', 'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004',
          'shift_jisx0213', 'utf_32', 'utf_32_be', 'utf_32_le', 'utf_16', 'utf_16_be', 'utf_16_le', 'utf_7', 'utf_8', 'utf_8_sig']

class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        '''
        creates an instance for Data Preparation class, it prepares the input data 
        for the embedding transformation 
        
        ## Parameters:
        
        config: DataPreparationConfig
            configuration for data preparation 
        '''
        self.config = config
        
    def prepare_data(self):
        '''
        reads the dataset from the local storage, splits the data in stratified sample 
        and stores the data in the local disk
        '''
        text_col = self.config.text_col
        target_col = self.config.target_col
        for codec in CODECS:
            try:
                dataset = pd.read_csv(self.config.pre_process_file, 
                                    encoding=codec, 
                                    sep=',', 
                                    names=self.config.col_names)
            except:
                continue
        
        logger.info(f"Loaded the dataset from {self.config.pre_process_file} using encoding - {codec}")
        
        dataset = dataset[[text_col, target_col]]
        dataset[target_col] = dataset[target_col].replace(4, 1)
        
        train_df, test_df = train_test_split(dataset.sample(frac=1), 
                                             test_size=0.3, 
                                             stratify=dataset[target_col], 
                                             random_state=42)
        valid_df, test_df = train_test_split(test_df.sample(frac=1), 
                                             test_size=0.5, 
                                             stratify=test_df[target_col], 
                                             random_state=42)
        
        train_df.to_csv(os.path.join(self.config.data_path, "train.csv"))
        valid_df.to_csv(os.path.join(self.config.data_path, "valid.csv"))
        test_df.to_csv(os.path.join(self.config.data_path, "test.csv"))
        
        logger.info(f"Data successfully splitted and saved at - {self.config.data_path}")
        