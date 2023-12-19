from sentify.config.configuration import ModelPredictionConfig

from transformers import AutoTokenizer, AutoModel
from gensim.models.fasttext import FastText
from gensim.utils import simple_preprocess
from gensim.test.utils import get_tmpfile
import tensorflow as tf 
import numpy as np 
import torch 
import json 
import os 


class ModelPrediction:
    def __init__(self, config: ModelPredictionConfig):
        '''
        creates an instance for prediction using trained model on text sequence
        '''
        self.config = config
        
    def predict(self, input_sequences: list[str]):
        '''
        predict the target label using the defined model
        
        ## Parameters:
        
        input_text: list[str]
            the sequences to be processed and predicted on 
            
        ## Returns:

        prediction: list[float]
            the predictions in likelihood for positive or negative sentiment
        '''
        if "fasttext" in self.config.active_model:
            embeddings = self.__get_fasttext_embeddings(input_sequences)
            
        elif "bert" in self.config.active_model:
            embeddings = self.__get_bert_embeddings(input_sequences)
            
        else:
            embeddings_fasttext = self.__get_fasttext_embeddings(input_sequences)
            embeddings_bert = self.__get_bert_embeddings(input_sequences)
            
            embeddings = [embeddings_bert, embeddings_fasttext]
        
        model = self.__load_prediction_model()
        
        predictions = model.predict(embeddings)
        
        return predictions
            
    def __get_bert_embeddings(self, input_sequences: list[str]):
        tokenizer, model = self.__load_bert_model()
        
        embeddings_merged = None
        embeddings_hidden = []
        for seq in input_sequences:
            tokens = tokenizer([seq], return_tensors='pt', padding="max_length",
                                truncation=True, max_length=self.config.bert_seq_len)

            with torch.no_grad():
                outputs = model(**tokens)

            embeddings = outputs.last_hidden_state
            embeddings_hidden.append(embeddings[0])
            cls_embeddings = embeddings[:, 0, :].numpy()
            if embeddings_merged is None:
                embeddings_merged = cls_embeddings
            else:
                embeddings_merged = np.append(
                    embeddings_merged, cls_embeddings, axis=0)
        
        if "dense" in self.config.active_model:
            return embeddings_merged

        return np.array(embeddings_hidden)
    
    def __get_fasttext_embeddings(self, input_sequences: list[str]):
        tokenizer, embed_model = self.__load_fasttext_model()
            
        if "dense" in self.config.active_model:
            embeddings = []
            for seq in input_sequences:
                processed_seq = simple_preprocess(seq)
                if processed_seq == []:
                    seq_embedding = np.zeros((300, ))
                else:
                    word_vectors = [embed_model.wv[word] 
                                    for word in processed_seq if word in embed_model.wv]
                    seq_embedding = np.mean(word_vectors, axis=0)
                
                embeddings.append(seq_embedding)
            embeddings = np.array(embeddings)
        else:
            embeddings = tokenizer.texts_to_sequences(input_sequences)
            embeddings = tf.keras.preprocessing.sequence.pad_sequences(
                embeddings, maxlen=self.config.fasttext_seq_len
            )
        
        return embeddings
            
    def __load_fasttext_model(self):
        # fname = get_tmpfile(self.config.fasttext_model)
        model = FastText.load(self.config.fasttext_model)
        
        with open(self.config.fasttext_tokenizer, 'r') as file:
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
                json.load(file)
            )
        
        return tokenizer, model 
    
    def __load_bert_model(self):
        if os.path.exists(self.config.bert_model) \
            and os.path.exists(self.config.bert_tokenizer):
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.bert_tokenizer
            )
            model = AutoModel.from_pretrained(
                self.config.bert_model
            )
        else: 
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name
            )
            model = AutoModel.from_pretrained(
                self.config.model_name
            )
            
        return tokenizer, model 
    
    def __load_prediction_model(self):
        model = tf.keras.models.load_model(
            os.path.join(self.config.models_path, self.config.active_model+".keras")
        )
        
        return model 
    