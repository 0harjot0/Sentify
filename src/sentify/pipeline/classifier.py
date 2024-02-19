from transformers import pipeline

class Classifier:
    def __init__(self):
        self.classifier_emotions = pipeline(
            task="text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None 
        )
        self.classifier_sentiments = pipeline(
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            return_all_scores=True
        )
    
    def predict_emotions(self, sequences):
        outputs = self.classifier_emotions(sequences)
        
        return outputs 
    
    def predict_sentiments(self, sequences):
        results = self.classifier_sentiments(sequences)
        
        return results
