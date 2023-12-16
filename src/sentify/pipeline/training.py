from sentify.config.configuration import ConfigurationManager
from sentify.components.data_ingestion import DataIngestion
from sentify.components.data_validation import DataValidation
from sentify.components.data_preparation import DataPreparation
from sentify.components.data_transformation import DataTransformation
from sentify.components.model_trainer import ModelTrainer
from logger import logger


class TrainingPipeline:
    def __init__(self):
        '''
        creates an instance of the Training Pipeline for Sentiment Analysis
        '''
        self.config = ConfigurationManager()
        
    def prepare_pipeline(self):
        try:
            logger.info("Training Pipeline Started!!!")
            self.__data_ingestion()
            
            self.__data_preparation_validation()
            
            logger.info("Completed Data and Training Pipeline Preparation")
        except Exception as e:
            logger.exception(e)
            raise e 
        
    def train_model(self, model_name: str, embed_type: str, **kwargs):
        try:
            logger.info("Started Model training and Data Transformation ")
            
            self.__data_transformation(embed_type)
            
            model_trainer_config = self.config.get_model_trainer_config()
            model_trainer = ModelTrainer(model_trainer_config)
            scores = None 
            
            if model_name == "dense":
                scores = model_trainer.train_dense_model(embed_type, **kwargs)
            elif model_name == "lstm":
                scores = model_trainer.train_lstm_model(embed_type, **kwargs)
            elif model_name == "bilstm":
                scores = model_trainer.train_bilstm_model(embed_type, **kwargs)
            elif model_name == "cnn":
                scores = model_trainer.train_cnn_model(embed_type, **kwargs)
            else:
                logger.info("Invalid model_name {}".format(model_name))
                
            logger.info("Model Training Completed!")
            
            return scores 
        except Exception as e:
            logger.exception(e)
            raise e 
            
    def __data_ingestion(self):
        logger.info("Starting Data Ingestion")
        data_ingestion_config = self.config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_data()
        data_ingestion.extract_zip_data()
        logger.info("Data Ingestion Finished")
    
    def __data_preparation_validation(self):
        logger.info("Validating and Preparing Data")
        data_validation_config = self.config.get_data_validation_config()
        data_preparation_config = self.config.get_data_preparation_config()
        
        data_validation = DataValidation(data_validation_config)
        data_validation.pre_process_validation()
        
        data_preparation = DataPreparation(data_preparation_config)
        data_preparation.prepare_data()
        
        data_validation.post_process_validation()
        logger.info("Validation and Preparation Completed")
        
    def __data_transformation(self, model_type: str):
        logger.info("Data Transformation Started")
        data_transformation_config = self.config.get_data_transformation_config()
        data_transformation = DataTransformation(data_transformation_config)
        if model_type == 'bert':
            logger.info("Preparing Embeddings for BERT")
            data_transformation.prepare_bert_embeddings()
        elif model_type == 'fasttext':
            logger.info("Preparing Embeddings for Fasttext")
            data_transformation.prepare_fasttext_embeddings()
        elif model_type == 'combine':
            logger.info("Preparing Embeddings for Fasttext")
            data_transformation.prepare_fasttext_embeddings()
            logger.info("Preparing Embeddings for BERT")
            data_transformation.prepare_bert_embeddings()

        logger.info("Data Transformation Finished")
    