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

# def extract_audio_features(filepath):
#     """
#     Extracts a comprehensive set of audio features from a given audio file.
    
#     Parameters:
#     filepath (str or Path): Path to the audio file.

#     Returns:
#     np.ndarray: Concatenated feature array for the given audio file.
#     """
#     y, sr = librosa.load(filepath, sr=None)
    
#     # Extracting audio features
#     zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y).T
#     root_mean_square_energy = librosa.feature.rms(y=y).T
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
#     spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).T
#     spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).T
#     spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr,fmin=200.0, n_bands=6).T
#     spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).T
#     mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000).T

#     # Concatenating all features into a single array
#     features = np.concatenate(
#         (
#             mfccs, mel_spectrogram, zero_crossing_rate, root_mean_square_energy,
#             spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_rolloff
#         ), axis=1
#     )
    
#     return features


# def generate_train_test_split(train_size=8):
#     """
#     Generates a train-test split of folds.

#     Parameters:
#     train_size (int): Number of folds to be used for training (default is 8).

#     Returns:
#     tuple: Lists containing the training and testing folds.
#     """
#     all_folds = list(range(1, 11))  # Folds numbered from 1 to 10
#     train_folds = random.sample(all_folds, train_size)
#     test_folds = [fold for fold in all_folds if fold not in train_folds]
    
#     return train_folds, test_folds


# def create_dataset(audio_directory, metadata_df, train_folds, test_folds):
#     """
#     Creates the dataset for training and testing neural network models.
    
#     Parameters:
#     audio_directory (str or Path): Directory containing the audio files.
#     metadata_df (DataFrame): DataFrame containing metadata for the audio files (must include 'slice_file_name', 'fold', 'classID').
#     train_folds (list): List of fold numbers to be used for training.
#     test_folds (list): List of fold numbers to be used for testing.
    
#     Returns:
#     tuple: Scaled feature arrays and corresponding labels for training and testing datasets.
#     """
#     audio_directory = Path(audio_directory)
#     train_features, test_features = [], []
#     y_train, y_test = [], []

#     # Extract features for each audio file
#     for file in audio_directory.rglob("*.wav"):
#         fold = metadata_df.loc[metadata_df['slice_file_name'] == file.name, 'fold'].iloc[0]
#         features = extract_audio_features(file)

#         if fold in train_folds:
#             y_train.append(metadata_df.loc[metadata_df['slice_file_name'] == file.name, 'class'].iloc[0])
#             train_features.append(features)
#         else:
#             y_test.append(metadata_df.loc[metadata_df['slice_file_name'] == file.name, 'class'].iloc[0])
#             test_features.append(features)
    
#     # Scaling features
#     train_features_combined = np.vstack(train_features)
#     scaler = StandardScaler()
#     scaler.fit(train_features_combined)

#     X_train = [scaler.transform(features) for features in train_features]
#     X_test = [scaler.transform(features) for features in test_features]

#     return X_train, X_test, y_train, y_test
