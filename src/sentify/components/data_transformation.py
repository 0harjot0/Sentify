from sentify.config.configuration import DataTransformationConfig
from utils.common import create_directories, check_files_exists
from logger import logger

from transformers import AutoTokenizer, AutoModel
from gensim.models.fasttext import FastText
from gensim.utils import simple_preprocess
from gensim.test.utils import get_tmpfile
import tensorflow as tf 
import pandas as pd 
import numpy as np 
import torch
import json 
import os 

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        '''
        creates an instance for Data Transformation class, it transforms 
        the input textual data into embeddings 
        
        ## Parameters:
        
        config: DataTransformationConfig
            configuration for Data Transformation
        '''
        self.config = config 
        self.__prepare_dataset()
        self.__save_labels()
        
    def prepare_bert_embeddings(self):
        '''
        prepares the embeddings with the mBERT for defined parameters
        '''
        if check_files_exists([
            
        ]):
            logger.info("BERT Embeddings and Model already prepared.")
            return 
        
        self.bert_tokenizer, self.bert_model = self.__load_bert_model()
        
        self.X_train_bert_dense_embeds, self.X_train_bert_lstm_embeds = self.__get_bert_embeddings(
            self.X_train_raw
        )

        self.X_valid_bert_dense_embeds, self.X_valid_bert_lstm_embeds = self.__get_bert_embeddings(
            self.X_valid_raw
        )

        self.X_test_bert_dense_embeds, self.X_test_bert_lstm_embeds = self.__get_bert_embeddings(
            self.X_test_raw
        )

        logger.info("mBERT Embeddings prepared.")
        
        self.__save_bert_embeddings()
        
    def __save_bert_embeddings(self):
        create_directories(self.config.bert_embedding)

        np.savez(self.config.bert_embedding,
                 train_bert_dense=self.X_train_bert_dense_embeds, train_bert_lstm=self.X_train_bert_lstm_embeds,
                 valid_bert_dense=self.X_valid_bert_dense_embeds, valid_bert_lstm=self.X_valid_bert_lstm_embeds,
                 test_bert_dense=self.X_test_bert_dense_embeds, test_bert_lstm=self.X_test_bert_lstm_embeds)

        print("Embeddings saved at path - {}".format(self.config.bert_embedding))
        
    def __get_bert_embeddings(self, sequences):
        embeddings_merged = None
        embeddings_hidden = []
        count = 0
        for seq in sequences:
            tokens = self.bert_tokenizer([seq], return_tensors='pt', padding="max_length",
                                         truncation=True, max_length=self.MAX_SEQ_LENGTH)

            with torch.no_grad():
                outputs = self.bert_model(**tokens)

            embeddings = outputs.last_hidden_state
            embeddings_hidden.append(embeddings[0])
            cls_embeddings = embeddings[:, 0, :].numpy()
            if embeddings_merged is None:
                embeddings_merged = cls_embeddings
            else:
                embeddings_merged = np.append(
                    embeddings_merged, cls_embeddings, axis=0)

            count += 1
            if count % 100 == 0:
                logger.info(str(count, embeddings_merged.shape))
        
        return embeddings_merged, np.array(embeddings_hidden)
    
    def __load_bert_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModel.from_pretrained(self.config.model_name)
        
        return tokenizer, model
        
    def prepare_fasttext_embeddings(self):
        '''
        prepares the embeddings with the fasttext with defined parameters
        '''
        if check_files_exists([
            self.config.fasttext_embedding, 
            self.config.fasttext_embed_matrix, 
            self.config.fasttext_model, 
            self.config.fasttext_tokenizer
        ]):
            logger.info("Fasttext Embeddings and Model already prepared")
            return 
        
        self.fasttext_model, self.keras_tokenizer = self.__load_fasttext_model()        
        train_sequences = [simple_preprocess(seq) for seq in self.X_train_raw]
        
        self.fasttext_model.build_vocab(train_sequences)
        self.fasttext_model.train(train_sequences, 
                    total_examples=len(train_sequences), 
                    epochs=self.config.fasttext_epochs)
        self.keras_tokenizer.fit_on_texts(train_sequences)
        
        self.embedding_matrix = np.zeros((len(self.keras_tokenizer.word_index)+1, 
                                     self.fasttext_model.vector_size))
        
        for word, i in self.keras_tokenizer.word_index.items():
            try:
                self.embedding_matrix[i] = self.fasttext_model.wv[word]
            except:
                continue
        
        logger.info("Embeddings matrix successfully extracted.")

        self.X_train_fasttext_dense_embeds, self.X_train_fasttext_lstm_embeds = self.__get_fasttext_embeddings(
            self.X_train_raw
        )
        self.X_valid_fasttext_dense_embeds, self.X_valid_fasttext_lstm_embeds = self.__get_fasttext_embeddings(
            self.X_valid_raw
        )
        self.X_test_fasttext_dense_embeds, self.X_test_fasttext_lstm_embeds = self.__get_fasttext_embeddings(
            self.X_test_raw
        )

        logger.info("Fasttext Embeddings successfully prepared.")
        
        self.__save_fasttext_embeddings()
        self.__save_fasttext_models()
        
    def __save_fasttext_models(self):
        fname = get_tmpfile(self.config.fasttext_model)
        self.fasttext_model.save(fname)
        
        logger.info("Fasttext model saved at - {}".format(self.config.fasttext_model))
        
        with open(self.config.fasttext_tokenizer, "w") as file:
            json.dump(self.keras_tokenizer.to_json(), 
                      file)
        
        logger.info("Fasttext tokenizer saved at - {}".format(self.config.fasttext_tokenizer))
        
    def __save_fasttext_embeddings(self):
        create_directories(self.config.fasttext_embedding)

        np.savez(self.config.fasttext_embedding,
                 train_fasttext_dense=self.X_train_fasttext_dense_embeds, train_fasttext_lstm=self.X_train_fasttext_lstm_embeds,
                 valid_fasttext_dense=self.X_valid_fasttext_dense_embeds, valid_fasttext_lstm=self.X_valid_fasttext_lstm_embeds,
                 test_fasttext_dense=self.X_test_fasttext_dense_embeds, test_fasttext_lstm=self.X_test_fasttext_lstm_embeds)

        np.save(self.config.fasttext_embed_matrix,
                self.embedding_matrix)

        logger.info("Embeddings saved at path - {}".format(self.config.fasttext_embedding))
        
    def __get_fasttext_embeddings(self, sequences):
        embeddings_fasttext = []
        for seq in sequences:
            processed_seq = simple_preprocess(seq)
            if processed_seq == []:
                seq_embedding = np.zeros((self.config.vector_size, ))
            else:
                word_vectors = [self.fasttext_model.wv[word]
                                for word in processed_seq if word in self.fasttext_model.wv]
                seq_embedding = np.mean(word_vectors, axis=0)

            embeddings_fasttext.append(seq_embedding)

        embeddings_keras = self.keras_tokenizer.texts_to_sequences(sequences)
        embeddings_keras = tf.keras.preprocessing.sequence.pad_sequences(embeddings_keras,
                                                                         maxlen=self.config.fasttext_seq_len)
        
        return np.array(embeddings_fasttext), embeddings_keras
        
    def __load_fasttext_model(self):
        model = FastText(vector_size=self.config.vector_size, 
                         window=5, min_count=5, alpha=0.025, workers=8)
        
        tokenizer = tf.keras.preprocessing.text.Tokenizer(self.config.word_count,
                                                          lower=True)
        
        return model, tokenizer
        
    def __prepare_dataset(self):
        raw_train = pd.read_csv(self.config.data_path + "train.csv")
        self.X_train_raw = raw_train.iloc[:, 0].tolist()
        self.y_train = raw_train.iloc[:, 1].to_numpy()
        logger.info("Train data shape - ", self.X_train_raw.shape)
        
        raw_valid = pd.read_csv(self.config.data_path + "valid.csv")
        self.X_valid_raw = raw_valid.iloc[:, 0].tolist()
        self.y_valid = raw_valid.iloc[:, 1].to_numpy()
        logger.info("Validation data shape - ", self.X_valid_raw.shape)
        
        raw_test = pd.read_csv(self.config.data_path + "test.csv")
        self.X_test_raw = raw_test.iloc[:, 0].tolist()
        self.y_test = raw_test.iloc[:, 1].to_numpy()
        logger.info("Test data shape - ", self.X_test_raw.shape)
        
    def __save_labels(self):
        '''
        saves the labels from the prepared dataset in the defined path
        '''
        create_directories(self.config.labels)
        
        np.savez(self.config.labels,
                 train=self.y_train, test=self.y_test,
                 valid=self.y_valid)

        print("Labels saved at path - {}".format(self.config.labels))
        