import librosa
import numpy as np 
import pandas as pd 
import random
import os

def get_features(direction):

    y,sr=librosa.load(direction)

    zcr = librosa.feature.zero_crossing_rate(y=y)

    rmse = librosa.feature.rms(y=y)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    melspectrogram =librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128) 
    
    features= np.hstack([mfccs,melspectrogram,zcr,rmse,spectral_centroid,spectral_bandwidth,spectral_contrast,spectral_rolloff])
    return features

def generate_train_set(train=8):
    # Generar una lista de números del 1 al 10
    numberoffolds = list(range(1, 11))
    
    # Seleccionar aleatoriamente 8 números sin repetición
    lista1 = random.sample(numberoffolds, train)
    
    # Encontrar los números que no están en lista1
    lista2 = [num for num in numberoffolds if num not in lista1]

    train=[f'fold{x}'for x in lista1]
    test=[f'fold{x}'for x in lista2]
    
    return train, test


def create_dataset(directory_audio,df): 
    '''
    Creates a dataset for the NN Models
    '''

    id=[]
    fold=[]
    features=[]
    labels=[]
    for file in directory_audio.rglob("*.wav"):


            id.append(df[df['slice_file_name'] == file.name].loc['slice_file_name'])
            fold.append(df[df['slice_file_name'] == file.name].loc['fold'])
            labels.append(df[df['slice_file_name'] == file.name].loc['class'])
            
            #Getting the features via the get_features function  
            feat=get_features()
            features.append(df[df['slice_file_name'] == file])

            

    df=pd.concat([id,fold,features,labels],axis=1)
    return df