import os 
import numpy as np 
import pickle 
import librosa 
from math import ceil
import keras as k 


testPath = "D:/WorkSpace/Projects/Sentify/artifacts/Audio"
scalerPath = "D:/WorkSpace/Projects/Sentify/artifacts/abc.pkl"
modelPath = "D:/WorkSpace/Projects/Sentify/artifacts/best_model.hdf5"


class AudioSentiment:
    def __init__(self, test_path: str = testPath, 
                 scaler_path: str = scalerPath, model_path:str =  modelPath, 
                 factor: int = 2):
        self.test_path = test_path 
        self.scaler_path = scaler_path
        self.model_path = model_path
        self.sample_rate = 44100
        self.factor = factor
    
    def predict(self, filepath: str):
        test_data, _ = self.__scanFeatures(filepath, 128)

        with open(self.scaler_path, "rb") as file:
            scaler = pickle.load(file)
        scaler.clip = False

        test_data = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)

        predict = k.models.load_model(modelPath).predict(test_data)
        
        return predict
        
    def __scanFeatures(self, path, avgFeat):
        features = []

        # files = sorted(os.listdir(path))
        print("Scanning", path)

        X, _ = librosa.load(path,
                            res_type='kaiser_fast', offset=0.5, 
                            sr=self.sample_rate)

        for i in range(ceil(len(X) / (self.factor*self.sample_rate))):
            f = librosa.feature.melspectrogram(y=X[self.factor*self.sample_rate*i:self.factor*self.sample_rate*(i+1)], 
                                               sr=self.sample_rate)
            f = librosa.amplitude_to_db(f, ref=np.max)

            features.append(f)

        feat_mat = np.zeros((ceil(len(X) / (self.factor*self.sample_rate)), f.shape[0], avgFeat))
        for i, x in enumerate(features):
            xWidth = min(x.shape[1],avgFeat)
            feat_mat[i, :, :xWidth] = x[:,:xWidth]
        return feat_mat, path