import librosa
import numpy as np 
import pandas as pd 
import random
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
    """
    Generates a train-test split of folds.

    Parameters:
    train_size (int): Number of folds to be used for training (default is 8).

    Returns:
    tuple: Lists containing the training and testing folds.
    """
    all_folds = list(range(1, 11))  # Folds numbered from 1 to 10
    train_folds = random.sample(all_folds, train)
    test_folds = [fold for fold in all_folds if fold not in train_folds]
    
    return train_folds, test_folds


def create_dataset(directory_audio,df,train,test): 
    """
    Creates the dataset for training and testing neural network models.
    
    Parameters:
    audio_directory (str or Path): Directory containing the audio files.
    metadata_df (DataFrame): DataFrame containing metadata for the audio files (must include 'slice_file_name', 'fold', 'classID').
    train_folds (list): List of fold numbers to be used for training.
    test_folds (list): List of fold numbers to be used for testing.
    
    Returns:
    tuple: Scaled feature arrays and corresponding labels for training and testing datasets.
    """

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
                y_train.append(df.loc[df['slice_file_name'] == file.name, 'class'])
                train_features.append(feat)
            else: 
                y_test.append(df.loc[df['slice_file_name'] == file.name, 'class'])
                test_features.append(feat)
    
    # Antes del escalamiento
    plt.hist(train_features[:, 0], bins=20)  # Selecciona una característica para visualizar
    plt.title('Distribución antes del escalamiento')
    plt.show()

    
    train_features_com=np.vstack(train_features)
    scaler= StandardScaler() 
    scaler.fit(train_features_com) 

    X_train = [scaler.transform(features) for features in train_features]
    X_test = [scaler.transform(features) for features in test_features]

    return X_train,X_test,y_train,y_test


def cut_audio(audio_path, start, end, sr=None):
    """
    Cuts a segment from an audio file based on start and end times.

    Parameters:
    - audio_path (str): Path to the audio file.
    - start (float): Start time in seconds.
    - end (float): End time in seconds.
    - sr (int, optional): Sampling rate for loading the audio. Default is None (uses original sampling rate).

    Returns:
    - segment (np.ndarray): Audio segment between start and end times.
    - sr (int): Sampling rate of the audio.
    """
    # Load the full audio
    audio, sr = librosa.load(audio_path, sr=sr)
    
    # Calculate start and end frames
    start_frame = int(start * sr)
    end_frame = int(end * sr)
    
    # Cut the audio segment
    segment = audio[start_frame:end_frame]
    
    return segment, sr