from sentify.config.configuration import ModelTrainerConfig
from utils.common import  create_directories
from logger import logger

from tensorflow.keras.layers import Dense, Input, Conv2D, LSTM, Bidirectional
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from gensim.models.fasttext import FastText
from gensim.test.utils import get_tmpfile
import tensorflow as tf
from math import ceil
import pandas as pd
import numpy as np
import torch
import json 
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        '''
        creates model trainer class for designing, training and evaluating 
        the model on defined set of parameters
        
        ## Parameters:
        
        config: ModelTrainerConfig
            the configuration including parameters for models
        '''
        self.config = config
        
        self.bert_loaded = False 
        self.fasttext_loaded = False 
        
        self.train_history = {}
        
        self.activation = self.config.activation
        self.metrics = self.config.metrics
        self.optimizer = self.config.optimizer
        
        self.MODEL_COMBS = ['bert', 'fasttext', 'combine']
        self.MODEL_TYPE = ['dense', 'lstm', 'bilstm', 'cnn']    
    
    def __load_fasttext_embedding(self):
        numpy_file = np.load(self.config.fasttext_embedding)

        X_train_dense = numpy_file['train_fasttext_dense']
        X_train_lstm = numpy_file['train_fasttext_lstm']
        X_valid_dense = numpy_file['valid_fasttext_dense']
        X_valid_lstm = numpy_file['valid_fasttext_lstm']
        X_test_dense = numpy_file['test_fasttext_dense']
        X_test_lstm = numpy_file['test_fasttext_lstm']

        embedding_matrix = np.load(self.config.fasttext_embed_matrix)

        logger.info("Fasttext Embeddings successfully loaded.")
    
        return (
            X_train_dense, 
            X_train_lstm, 
            X_valid_dense, 
            X_valid_lstm, 
            X_test_dense, 
            X_test_lstm, 
            embedding_matrix
        )
    
    def __load_fasttext_model(self):
        # fname = get_tmpfile(self.config.fasttext_model)
        model = FastText.load(self.config.fasttext_model)
        
        with open(self.config.fasttext_tokenizer, "r") as file:
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
                json.load(file)
            )
            
        return model, tokenizer
        
    def __load_bert_embedding(self):
        numpy_file = np.load(self.config.bert_embedding)
        
        X_train_dense = numpy_file['train_bert_dense']
        X_train_lstm = numpy_file['train_bert_lstm']
        X_valid_dense = numpy_file['valid_bert_dense']
        X_valid_lstm = numpy_file['valid_bert_lstm']
        X_test_dense = numpy_file['test_bert_dense']
        X_test_lstm = numpy_file['test_bert_lstm']
        
        logger.info("BERT Embeddings successfully loaded.")
        
        return (
            X_train_dense, 
            X_train_lstm, 
            X_valid_dense, 
            X_valid_lstm, 
            X_test_dense, 
            X_test_lstm
        )
    
    def __load_labels(self):
        numpy_file = np.load(self.config.labels)
        
        y_train = numpy_file['train']
        y_test = numpy_file['test']
        y_valid = numpy_file['valid']
        
        return (
            y_train, 
            y_valid, 
            y_test
        ) 
        
    def __save_model(self, model, filename: str):
        create_directories([self.config.models_path])
        
        save_file = os.path.join(self.config.models_path, filename+".keras")
        model.save(save_file)
        
    def __evaluate_model(self, y_test, y_pred):
        try:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)

            logger.info(f"Accuracy - {accuracy}")
            logger.info(f"Precision - {precision}")
            logger.info(f"Recall - {recall}")
            logger.info(f"F1 - {f1}")
            logger.info(f"Roc-Auc - {roc_auc}")

            return accuracy, precision, recall, f1, roc_auc
        except Exception as e:
            logger.exception(e)
        
    def train_dense_model(self, model_type: str, layers: int = 3, units: int = 200):
        if model_type in self.MODEL_COMBS:
            if model_type == "bert":
                X_train, _, X_valid, _, X_test, _ = self.__load_bert_embedding()
                y_train, y_valid, y_test = self.__load_labels()
                
                model = tf.keras.models.Sequential()
                model.add(Input(shape=(768, )))
                for _ in range(layers):
                    model.add(Dense(units, activation=self.activation))
                model.add(Dense(1, activation='sigmoid'))

                model.compile(loss='binary_crossentropy',
                                              optimizer=self.optimizer,
                                              metrics=self.metrics)

                logger.info("Model compiled with summary ----- ")
                logger.info(model.summary())

                self.train_history['bert_dense'] = model.fit(X_train,
                                                             y_train,
                                                             epochs=self.config.epochs,
                                                             validation_data=(
                                                                 X_valid,
                                                                 y_valid
                                                            ))

                self.__save_model(model, 'bert_dense')
                scores = self.__evaluate_model(y_test, 
                                               np.round(model.predict(X_test)))

            elif model_type == "fasttext":
                X_train, _, X_valid, _, X_test, _, _ = self.__load_fasttext_embedding()
                y_train, y_valid, y_test = self.__load_labels()
                
                model = tf.keras.models.Sequential()
                model.add(Input(shape=(self.config.vector_size, )))
                for _ in range(layers):
                    model.add(
                        Dense(units, activation=self.activation))
                model.add(Dense(1, activation='sigmoid'))

                model.compile(loss='binary_crossentropy',
                              optimizer=self.optimizer,
                              metrics=self.metrics)

                logger.info("Model compiled with summary ----- ")
                logger.info(model.summary())

                self.train_history['fasttext_dense'] = model.fit(X_train,
                                                                 y_train,
                                                                 epochs=self.config.epochs,
                                                                 validation_data=(
                                                                     X_valid,
                                                                     y_valid
                                                                ))
                
                self.__save_model(model, 'fasttext_dense')
                scores = self.__evaluate_model(y_test, 
                                               np.round(model.predict(X_test)))

            else:
                X_train_bert, _, X_valid_bert, _, X_test_bert, _ = self.__load_bert_embedding()
                y_train, y_valid, y_test = self.__load_labels()
                
                X_train_fasttext, _, X_valid_fasttext, _, X_test_fasttext, _, _ = self.__load_bert_embedding()
                
                upper_layers = ceil(layers/2)

                dense_bert_input = Input(shape=(768, ))
                dense_bert_output = Dense(
                    units, activation=self.activation)(dense_bert_input)
                for _ in range(upper_layers-1):
                    dense_bert_output = Dense(
                        units, activation=self.activation)(dense_bert_output)

                dense_fasttext_input = Input(shape=(self.config.vector_size, ))
                dense_fasttext_output = Dense(
                    units, activation=self.activation)(dense_fasttext_input)
                for _ in range(upper_layers-1):
                    dense_fasttext_output = Dense(
                        units, activation=self.activation)(dense_fasttext_output)

                merged_input = tf.keras.layers.concatenate(
                    [dense_bert_output, dense_fasttext_output])
                merged_output = Dense(units, activation=self.activation)(merged_input)
                for _ in range(layers - upper_layers - 1):
                    merged_output = Dense(
                        units, activation=self.activation)(merged_output)
                merged_output = Dense(1, activation='sigmoid')(merged_output)

                self.combine_dense_model = tf.keras.models.Model(inputs=[dense_bert_input, dense_fasttext_input],
                                                                 outputs=merged_output)
                self.combine_dense_model.compile(loss='binary_crossentropy',
                                                 optimizer=self.optimizer,
                                                 metrics=self.metrics)

                logger.info("Model compiles with summary ----- ")
                logger.info(self.combine_dense_model.summary())

                self.train_history['combine_dense'] = self.combine_dense_model.fit([X_train_bert, X_train_fasttext],
                                                                                   y_train,
                                                                                   epochs=self.config.epochs,
                                                                                   validation_data=(
                                                                                       [X_valid_bert, X_valid_fasttext],
                                                                                       y_valid
                                                                                   ))

                self.__save_model(model, 'combined_dense')
                scores = self.__evaluate_model(y_test, 
                                               np.round(model.predict([X_test_bert, X_test_fasttext])))

            
            return scores
        else:
            logger.info("Model Type not in defined Model Combinations!!!")

    def train_lstm_model(self, model_type: str, layers: int = 3, units: int = 64):
        if model_type in self.MODEL_COMBS:
            if model_type == "bert":
                _, X_train, _, X_valid, _, X_test = self.__load_bert_embedding()
                y_train, y_valid, y_test = self.__load_labels()
                
                model = tf.keras.models.Sequential()
                for i in range(layers-1):
                    model.add(
                        LSTM(units//2**i, return_sequences=True))
                model.add(LSTM(units//2**(layers-1)))
                model.add(Dense(1, activation='sigmoid'))

                model.compile(loss='binary_crossentropy',
                              optimizer=self.optimizer,
                              metrics=self.metrics)

                self.train_history['bert_lstm'] = model.fit(X_train,
                                                             y_train,
                                                             epochs=self.config.epochs,
                                                             validation_data=(
                                                                 X_valid,
                                                                 y_valid
                                                            ))

                logger.info("Model compiles with summary ----- ")
                logger.info(model.summary())
                
                self.__save_model(model, 'bert_lstm')
                scores = self.__evaluate_model(y_test, 
                                               np.round(model.predict(X_test)))

            elif model_type == "fasttext":
                _, X_train, _, X_valid, _, X_test, embed_matrix = self.__load_fasttext_embedding()
                y_train, y_valid, y_test = self.__load_labels()
                _, tokenizer = self.__load_fasttext_model()
                
                model = tf.keras.models.Sequential()
                model.add(
                    Input(shape=(self.config.fasttext_seq_len, )))
                model.add(
                    tf.keras.layers.Embedding(len(tokenizer.word_index)+1,
                                              self.config.vector_size,
                                              weights=[embed_matrix],
                                              trainable=False)
                )
                for i in range(layers-1):
                    model.add(
                        LSTM(units//2**i, return_sequences=True))
                model.add(LSTM(units//2**(layers-1)))
                model.add(Dense(1, activation='sigmoid'))

                model.compile(loss='binary_crossentropy',
                              optimizer=self.optimizer,
                              metrics=self.metrics)

                self.train_history['fasttext_lstm'] = model.fit(X_train,
                                                                y_train,
                                                                epochs=self.config.epochs,
                                                                validation_data=(
                                                                     X_valid,
                                                                     y_valid
                                                                ))

                logger.info("Model compiles with summary ----- ")
                logger.info(model.summary())
                
                self.__save_model(model, 'fasttext_lstm')
                scores = self.__evaluate_model(y_test, 
                                               np.round(model.predict(X_test)))

            else:
                _, X_train_bert, _, X_valid_bert, _, X_test_bert = self.__load_bert_embedding()
                y_train, y_valid, y_test = self.__load_labels()
                
                _, X_train_fasttext, _, X_valid_fasttext, _, X_test_fasttext, embed_matrix = self.__load_fasttext_embedding()
                _, tokenizer = self.__load_fasttext_model()
                
                upper_layers = ceil(layers/2)

                lstm_bert_input = Input(shape=(self.config.bert_seq_len, 768, ))
                lstm_fasttext_input = Input(shape=(self.config.fasttext_seq_len, ))

                lstm_bert_output = LSTM(
                    units, return_sequences=True)(lstm_bert_input)
                lstm_fasttext_embeds = tf.keras.layers.Embedding(len(tokenizer.word_index)+1,
                                                                 self.config.vector_size,
                                                                 weights=[
                                                                     embed_matrix],
                                                                 trainable=False)(lstm_fasttext_input)
                lstm_fasttext_output = LSTM(
                    units, return_sequences=True)(lstm_fasttext_embeds)
                for i in range(upper_layers-1):
                    lstm_bert_output = LSTM(
                        units//2**(i+1), return_sequences=True)(lstm_bert_output)
                    lstm_fasttext_output = LSTM(
                        units//2**(i+1), return_sequences=True)(lstm_fasttext_output)

                merged_input = tf.keras.layers.concatenate(
                    [lstm_bert_output, lstm_fasttext_output])
                if layers-upper_layers > 1:
                    merged_output = LSTM(
                        units//2**(upper_layers-1), return_sequences=True)(merged_input)
                    for i in range(layers-upper_layers-2):
                        merged_output = LSTM(
                            units//2**(upper_layers+i), return_sequences=True)(merged_output)
                    merged_output = LSTM(units//2**(layers-1))(merged_output)
                else:
                    merged_output = LSTM(units//2**(layers-1))(merged_input)

                merged_output = Dense(1, activation='sigmoid')(merged_output)

                self.combine_lstm_model = tf.keras.models.Model(inputs=[lstm_bert_input, lstm_fasttext_input],
                                                                outputs=merged_output)

                self.combine_lstm_model.compile(loss='binary_crossentropy',
                                                optimizer=self.optimizer,
                                                metrics=self.metrics)

                self.train_history['combine_lstm'] = self.combine_lstm_model.fit([X_train_bert, X_train_fasttext],
                                                                                 y_train,
                                                                                 epochs=self.config.epochs,
                                                                                 validation_data=(
                                                                                     [X_valid_bert, X_valid_fasttext],
                                                                                     y_valid
                ))

                logger.info("Model compiles with summary ----- ")
                logger.info(self.combine_lstm_model.summary())
                
                self.__save_model(model, 'combine_lstm')
                scores = self.__evaluate_model(y_test, 
                                               np.round(model.predict([X_test_bert, X_test_fasttext])))
            
            return scores

        else:
            logger.info("Model Type not in defined Model Combinations!!!")

    def train_bilstm_model(self, model_type: str, layers: int = 3, units: int = 64):
        if model_type in self.MODEL_COMBS:
            if model_type == "bert":
                _, X_train, _, X_valid, _, X_test = self.__load_bert_embedding()
                y_train, y_valid, y_test = self.__load_labels()
                
                model = tf.keras.models.Sequential()
                for i in range(layers-1):
                    model.add(Bidirectional(
                        LSTM(units//2**i, return_sequences=True)))
                model.add(
                    Bidirectional(LSTM(units//2**(layers-1))))
                model.add(Dense(1, activation='sigmoid'))

                model.compile(loss='binary_crossentropy',
                                               optimizer=self.optimizer,
                                               metrics=self.metrics)

                self.train_history['bert_bilstm'] = model.fit(X_train,
                                                             y_train,
                                                             epochs=self.config.epochs,
                                                             validation_data=(
                                                                 X_valid,
                                                                 y_valid
                                                            ))

                logger.info("Model compiles with summary ----- ")
                logger.info(model.summary())
                
                self.__save_model(model, 'bert_bilstm')
                scores = self.__evaluate_model(y_test, 
                                               np.round(model.predict(X_test)))

            elif model_type == "fasttext":
                _, X_train, _, X_valid, _, X_test, embed_matrix = self.__load_fasttext_embedding()
                y_train, y_valid, y_test = self.__load_labels()
                _, tokenizer = self.__load_fasttext_model()
                
                model = tf.keras.models.Sequential()
                model.add(
                    Input(shape=(self.config.fasttext_seq_len, )))
                model.add(
                    tf.keras.layers.Embedding(len(tokenizer.word_index)+1,
                                              self.config.vector_size,
                                              weights=[embed_matrix],
                                              trainable=False)
                )
                for i in range(layers-1):
                    model.add(Bidirectional(
                        LSTM(units//2**i, return_sequences=True)))
                model.add(
                    Bidirectional(LSTM(units//2**(layers-1))))
                model.add(Dense(1, activation='sigmoid'))

                model.compile(loss='binary_crossentropy',
                                optimizer=self.optimizer,
                                metrics=self.metrics)

                self.train_history['fasttext_bilstm'] = model.fit(X_train,
                                                                 y_train,
                                                                 epochs=self.config.epochs,
                                                                 validation_data=(
                                                                     X_valid,
                                                                     y_valid
                                                                ))

                logger.info("Model compiles with summary ----- ")
                logger.info(model.summary())
                
                self.__save_model(model, 'fasttext_bilstm')
                scores = self.__evaluate_model(y_test, 
                                               np.round(model.predict(X_test)))

            else:
                _, X_train_bert, _, X_valid_bert, _, X_test_bert = self.__load_bert_embedding()
                y_train, y_valid, y_test = self.__load_labels()
                
                _, X_train_fasttext, _, X_valid_fasttext, _, X_test_fasttext, embed_matrix = self.__load_fasttext_embedding()
                _, tokenizer = self.__load_fasttext_model()
                
                upper_layers = ceil(layers/2)

                bilstm_bert_input = Input(shape=(self.config.bert_seq_len, 768, ))
                bilstm_fasttext_input = Input(shape=(self.config.fasttext_seq_len, ))

                bilstm_bert_output = Bidirectional(
                    LSTM(units, return_sequences=True))(bilstm_bert_input)
                bilstm_fasttext_embeds = tf.keras.layers.Embedding(len(tokenizer.word_index)+1,
                                                                   self.config.vector_size,
                                                                   weights=[
                                                                       embed_matrix],
                                                                   trainable=False)(bilstm_fasttext_input)
                bilstm_fasttext_output = Bidirectional(
                    LSTM(units, return_sequences=True))(bilstm_fasttext_embeds)
                for i in range(upper_layers-1):
                    bilstm_bert_output = Bidirectional(
                        LSTM(units//2**(i+1), return_sequences=True))(bilstm_bert_output)
                    bilstm_fasttext_output = Bidirectional(
                        LSTM(units//2**(i+1), return_sequences=True))(bilstm_fasttext_output)

                merged_input = tf.keras.layers.concatenate(
                    [bilstm_bert_output, bilstm_fasttext_output])
                if layers-upper_layers > 1:
                    merged_output = Bidirectional(
                        LSTM(units//2**(upper_layers-1), return_sequences=True))(merged_input)
                    for i in range(layers-upper_layers-2):
                        merged_output = Bidirectional(
                            LSTM(units//2**(upper_layers+i), return_sequences=True))(merged_output)
                    merged_output = Bidirectional(
                        LSTM(units//2**(layers-1)))(merged_output)
                else:
                    merged_output = Bidirectional(
                        LSTM(units//2**(layers-1)))(merged_input)

                merged_output = Dense(1, activation='sigmoid')(merged_output)

                self.combine_bilstm_model = tf.keras.models.Model(inputs=[bilstm_bert_input, bilstm_fasttext_input],
                                                                  outputs=merged_output)

                self.combine_bilstm_model.compile(loss='binary_crossentropy',
                                                  optimizer=self.optimizer,
                                                  metrics=self.metrics)

                self.train_history['combine_bilstm'] = self.combine_bilstm_model.fit([X_train_bert, X_train_fasttext],
                                                                                   y_train,
                                                                                   epochs=self.config.epochs,
                                                                                   validation_data=(
                    [X_valid_bert, X_valid_fasttext],
                    y_valid
                ))

                logger.info("Model compiles with summary ----- ")
                logger.info(self.combine_bilstm_model.summary())
                
                self.__save_model(model, 'combine_bilstm')
                scores = self.__evaluate_model(y_test, 
                                               np.round(model.predict([X_test_bert, X_test_fasttext])))

            return scores 
        else:
            logger.info("Model Type not in defined Model Combinations!!!")

    def train_cnn_model(self, model_type: str, units: int = 512):
        if model_type in self.MODEL_COMBS:
            if model_type == 'bert':
                _, X_train, _, X_valid, _, X_test = self.__load_bert_embedding()
                y_train, y_valid, y_test = self.__load_labels()
                
                model = tf.keras.models.Sequential([
                    Input(shape=(self.config.bert_seq_len, 768, )),
                    tf.keras.layers.Reshape((self.config.bert_seq_len, 768, 1)),
                    Conv2D(units, kernel_size=(3, 768), padding='valid',
                           kernel_initializer="normal", activation=self.activation),
                    tf.keras.layers.MaxPool2D(pool_size=(self.config.bert_seq_len-3+1, 1),
                                              strides=(1, 1), padding='valid'),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])

                model.compile(loss='binary_crossentropy',
                                            optimizer=self.optimizer,
                                            metrics=self.metrics)

                logger.info("Model compiles with summary ----- ")
                logger.info(model.summary())

                self.train_history['bert_cnn'] = model.fit(X_train,
                                                            y_train,
                                                            epochs=self.config.epochs,
                                                            validation_data=(
                                                                X_valid,
                                                                y_valid
                                                            ))
                
                self.__save_model(model, 'bert_cnn')
                scores = self.__evaluate_model(y_test, 
                                               np.round(model.predict(X_test)))

            elif model_type == 'fasttext':
                # _, X_train, _, X_valid, _, X_test, embed_matrix = self.__load_fasttext_embedding()
                # y_train, y_valid, y_test = self.__load_labels()
                _, tokenizer = self.__load_fasttext_model()
                
                print(len(tokenizer.word_index))
                
                model = tf.keras.models.Sequential([
                    Input(shape=(self.config.fasttext_seq_len, )),
                    tf.keras.layers.Embedding(len(tokenizer.word_index)+1,
                                              self.config.vector_size,
                                              weights=[embed_matrix],
                                              trainable=False),
                    tf.keras.layers.Reshape(
                        (self.config.fasttext_seq_len, self.config.vector_size, 1)),
                    Conv2D(units, kernel_size=(3, self.config.vector_size), padding='valid',
                           kernel_initializer="normal", activation=self.activation),
                    tf.keras.layers.MaxPool2D(pool_size=(self.config.fasttext_seq_len-3+1, 1),
                                              strides=(1, 1), padding="valid"),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dropout(0.5),
                    Dense(1, activation='sigmoid')
                ])

                model.compile(loss='binary_crossentropy',
                                                optimizer=self.optimizer,
                                                metrics=self.metrics)

                logger.info("Model compiles with summary ----- ")
                logger.info(model.summary())

                self.train_history['fasttext_cnn'] = model.fit(X_train,
                                                                y_train,
                                                                epochs=self.config.epochs,
                                                                validation_data=(
                                                                    X_valid,
                                                                    y_valid
                                                                ))
                
                self.__save_model(model, 'fasttext_cnn')
                scores = self.__evaluate_model(y_test, 
                                               np.round(model.predict(X_test)))

            else:
                _, X_train_bert, _, X_valid_bert, _, X_test_bert = self.__load_bert_embedding()
                y_train, y_valid, y_test = self.__load_labels()
                
                _, X_train_fasttext, _, X_valid_fasttext, _, X_test_fasttext, embed_matrix = self.__load_fasttext_embedding()
                _, tokenizer = self.__load_fasttext_model()
                
                bert_cnn_input = Input(shape=(self.config.bert_seq_len, 768, ))
                bert_cnn_reshape = tf.keras.layers.Reshape(
                    (self.config.bert_seq_len, 768, 1))(bert_cnn_input)
                bert_cnn_main = Conv2D(units, (3, 768), padding='valid',
                                       kernel_initializer='normal', activation=self.activation)(bert_cnn_reshape)
                bert_cnn_pool = tf.keras.layers.MaxPool2D((self.config.bert_seq_len-3+1, 1),
                                                          strides=(1, 1), padding='valid')(bert_cnn_main)

                fasttext_cnn_input = Input(shape=(self.config.fasttext_seq_len, ))
                fasttext_cnn_embed = tf.keras.layers.Embedding(len(tokenizer.word_index)+1,
                                                               self.config.vector_size,
                                                               weights=[
                                                                   embed_matrix],
                                                               trainable=False)(fasttext_cnn_input)
                fasttext_cnn_reshape = tf.keras.layers.Reshape(
                    (self.config.fasttext_seq_len, self.config.vector_size, 1))(fasttext_cnn_embed)
                fasttext_cnn_main = Conv2D(units, (3, self.config.vector_size), padding='valid',
                                           kernel_initializer='normal', activation=self.activation)(fasttext_cnn_reshape)
                fasttext_cnn_pool = tf.keras.layers.MaxPool2D((self.config.fasttext_seq_len-3+1, 1),
                                                              strides=(1, 1),
                                                              padding='valid')(fasttext_cnn_main)

                merged_cnn_input = tf.keras.layers.Concatenate(
                    axis=1)([bert_cnn_pool, fasttext_cnn_pool])
                merged_cnn_flat = tf.keras.layers.Flatten()(merged_cnn_input)
                merged_cnn_drop = tf.keras.layers.Dropout(0.5)(merged_cnn_flat)
                merged_cnn_output = Dense(
                    1, activation='sigmoid')(merged_cnn_drop)

                self.combine_cnn_model = tf.keras.models.Model(inputs=[bert_cnn_input, fasttext_cnn_input],
                                                               outputs=merged_cnn_output)

                self.combine_cnn_model.compile(loss='binary_crossentropy',
                                               optimizer=self.optimizer,
                                               metrics=self.metrics)

                logger.info("Model compiles with summary ----- ")
                logger.info(self.combine_cnn_model.summary())

                self.train_history['combine_cnn'] = self.combine_cnn_model.fit([X_train_bert, X_train_fasttext],
                                                                               y_train,
                                                                               epochs=self.config.epochs,
                                                                               validation_data=(
                                                                                   [X_valid_bert, X_valid_fasttext],
                                                                                   y_valid
                ))
                
                self.__save_model(model, 'combine_cnn')
                scores = self.__evaluate_model(y_test, 
                                               np.round(model.predict([X_test_bert, X_test_fasttext])))
            
            return scores 

        else:
            logger.info("Model Type not in defined Model Combinations!!!")
