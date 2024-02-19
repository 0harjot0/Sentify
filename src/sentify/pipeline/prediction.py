from sentify.components.model_prediction import ModelPrediction
from sentify.config.configuration import ConfigurationManager
from logger import logger


class PredictionPipeline:
    def __init__(self):
        '''
        creates a prediction pipeline for predicting over given set of sequences
        '''
        self.config = ConfigurationManager()
        model_prediction_config = self.config.get_model_prediction_config()
        self.model_prediction = ModelPrediction(model_prediction_config, "combine_emotion_score.keras")
        self.model_prediction_1 = ModelPrediction(model_prediction_config, "combine_sentiment_score.keras")
    
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
            
            predictions = self.model_prediction.predict(input_sequences)
            predictions_1 = self.model_prediction_1.predict(input_sequences)
            
            logger.info("Prediction Completed")
            
            return {"Emotions": predictions, "Sentiments": predictions_1}
        except Exception as e:
            logger.exception(e)
            raise e 
        