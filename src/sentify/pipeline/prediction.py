from sentify.components.model_prediction import ModelPrediction
from sentify.config.configuration import ConfigurationManager
from logger import logger


class PredictionPipeline:
    def __init__(self):
        '''
        creates a prediction pipeline for predicting over given set of sequences
        '''
        self.config = ConfigurationManager()
    
    def predict(self, input_sequences: list[str]):
        '''
        predicts the likelihood of sentiments for the given set of sequences
        
        ## Parameters:
        
        input_sequences: list[str]
            the sequences to do the prediction on 
            
        ## Returns:
        
        predictions: list[float]
            the predictions with likelihood for each sequence
        '''
        try:
            logger.info("Prediction started")
            model_prediction_config = self.config.get_model_prediction_config()
            model_prediction = ModelPrediction(model_prediction_config)
            
            predictions = model_prediction.predict(input_sequences)
            
            logger.info("Prediction Completed")
            
            return predictions
        except Exception as e:
            logger.exception(e)
            raise e 
        