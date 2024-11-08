import librosa
import numpy as np 
import pandas as pd 
import random
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def get_features(direction):

    y,sr=librosa.load(direction)

    zcr = librosa.feature.zero_crossing_rate(y=y).T

    rmse = librosa.feature.rms(y=y).T

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
    
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).T

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).T

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).T

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).T

    melspectrogram =librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000).T

    features= np.concatenate((mfccs,melspectrogram,zcr,rmse,spectral_centroid,spectral_bandwidth,spectral_contrast,spectral_rolloff),axis=1)
    
    return features

def generate_train_set(train=8):
    # Generar una lista de números del 1 al 10, para mirar que fold nos quedamos 
    numberoffolds = list(range(1, 11))
    # Seleccionar aleatoriamente 8 números sin repetición para fold de train
    l_train = random.sample(numberoffolds, train)
    # Encontrar los números que no están en lista1
    l_test = [num for num in numberoffolds if num not in l_train]
    
    return l_train, l_test


def create_dataset(directory_audio,df,train,test): 
    '''
    Creates a dataset for the NN Models
    '''

    directory_audio=Path(directory_audio)
    train_features = []
    test_features = []

    X_train=[]
    X_test= []
    y_train=[]
    y_test=[]

    for file in directory_audio.rglob("*.wav"):
            fold=df.loc[df['slice_file_name'] == file.name, 'fold'].iloc[0]
            #Getting the features list via the get_features function  
            feat=get_features(file)
            if fold in train:
                y_train.append(df.loc[df['slice_file_name'] == file.name, 'classID'])
                train_features.append(feat)
            else: 
                y_test.append(df.loc[df['slice_file_name'] == file.name, 'classID'])
                test_features.append(feat)
    
    train_features_com=np.vstack(train_features)
    scaler= StandardScaler() 
    scaler.fit(train_features_com) 

    X_train = [scaler.transform(features) for features in train_features]
    X_test = [scaler.transform(features) for features in test_features]

    return X_train,X_test,y_train,y_test