import librosa
import numpy as np 
import pandas as pd 
import random
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis

def match_shapes(arrays):
    """Recorta o ajusta las matrices para que todas tengan el mismo número de filas."""
    min_length = min(arr.shape[0] for arr in arrays)
    return [arr[:min_length, :] for arr in arrays]

def get_features(direction,target_length = 170):
    
    y,sr=librosa.load(direction, sr=22050)
    
    # Definir parámetros de extracción
    frame_length = int(sr * 0.0232)  # 23.2 ms en muestras
    hop_length = frame_length // 2   # 50% de superposición

    zcr = librosa.feature.zero_crossing_rate(y=y).T

    rmse = librosa.feature.rms(y=y).T

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=25 , hop_length=hop_length, n_fft=frame_length).T

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).T

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).T

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).T

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).T

    # Extraer espectrograma de Mel (40 bandas de Mel entre 0 y 22050 Hz)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, hop_length=hop_length, n_fft=frame_length, fmax=22050).T
    
    # Ajustar el ancho de la delta en función del tamaño de mfccs, asegurando que sea impar y >= 3
    width = min(9, mfccs.shape[1])  # Valor máximo de 9 o el tamaño de los cuadros
    if width < 3:
        width = 3  # Mínimo permitido
    elif width % 2 == 0:
        width -= 1  # Asegurarse de que sea impar

    
    # Calcular las primeras y segundas derivadas de los MFCCs y sus estadísticas
    delta_mfccs = librosa.feature.delta(mfccs,width=width)
    delta2_mfccs = librosa.feature.delta(mfccs,width=width, order=2)

    features_list = [mfccs, mel_spectrogram, zcr, rmse, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_rolloff, delta_mfccs, delta2_mfccs]
    features_list = match_shapes(features_list)

    features = np.concatenate(features_list, axis=1)
    #print(features.shape)
    
    return features



def compute_statistics(feature_array):
    return {
        'min': np.min(feature_array, axis=1),
        'max': np.max(feature_array, axis=1),
        'mean': np.mean(feature_array, axis=1),
        'median': np.median(feature_array, axis=1),
        'variance': np.var(feature_array, axis=1),
        'skewness': skew(feature_array, axis=1, nan_policy='omit'),
        'kurtosis': kurtosis(feature_array, axis=1, nan_policy='omit')
    }

def get_features_salomon(direction):
    # Cargar el archivo de audio
    y, sr = librosa.load(direction, sr=22050)
    
    # Extraer MFCCs con librosa (25 coeficientes)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=25, hop_length=hop_length, n_fft=frame_length)
    
    # Extraer espectrograma de Mel (40 bandas de Mel entre 0 y 22050 Hz)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, hop_length=hop_length, n_fft=frame_length, fmax=22050)
    

    # Calcular estadísticas sobre los MFCCs
    mfcc_stats = compute_statistics(mfccs)
    
    # Calcular estadísticas sobre las bandas de Mel
    mel_stats = compute_statistics(mel_spectrogram)
    
    delta_stats = compute_statistics(delta_mfccs)
    delta2_stats = compute_statistics(delta2_mfccs)
    
    # Concatenar todas las estadísticas en un solo vector
    features = np.concatenate(
        (
        mfcc_stats['min'], mfcc_stats['max'], mfcc_stats['mean'], 
        mfcc_stats['median'], mfcc_stats['variance'], mfcc_stats['skewness'],
        mfcc_stats['kurtosis'], mel_stats['min'], mel_stats['max'],
        mel_stats['mean'], mel_stats['median'], mel_stats['variance'],
        mel_stats['skewness'], mel_stats['kurtosis'], delta_stats['mean'], 
        delta_stats['variance'], delta2_stats['mean'], delta2_stats['variance']
        )
    )
    
    # Retornar el vector de características
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


def create_folds(directory_audio, df):
    """
    This function creates a dataset by reading audio files from a specified directory, extracting their features, 
    and organizing them into different folds (10 in this case) based on the 'fold' information in the DataFrame 'df'.
    
    The function will return 10 variables (x1, y1, ..., x10, y10) corresponding to features and labels for each fold.
    
    Parameters:
    -----------
    directory_audio : str or Path
        The path to the directory containing audio (.wav) files.
        
    df : pandas.DataFrame
        A DataFrame containing metadata for the audio files. It should have at least two columns:
        'slice_file_name' (the filename of the audio file) and 'fold' (indicating which fold the file belongs to).
        
    Returns:
    --------
    x1, y1, x2, y2, ..., x10, y10 : 
        Features (x) and labels (y) for each fold (10 in total). Each x represents the feature matrix for fold i, 
        and each y represents the label vector for fold i.
    """
    
    directory_audio = Path(directory_audio)
    
    # Initialize lists to store features (x) and labels (y) for each of the 10 folds
    x_folds = [[] for _ in range(10)]  # Lists to hold features for each fold
    y_folds = [[] for _ in range(10)]  # Lists to hold labels for each fold

    # Iterate through each audio file in the specified directory (including subdirectories)
    for file in directory_audio.rglob("*.wav"):
        # Retrieve the fold number for the current file from the DataFrame
        fold = df.loc[df['slice_file_name'] == file.name, 'fold'].iloc[0]
        
        # Extract the features of the audio file using a predefined function `get_features_salomon`
        feat = get_features(file)
        
        # Check for NaN or Inf values in the extracted features
        if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
            #print(f"Warning: NaN or Inf found in features for {file.name}. Replacing with zeros.")
            feat = np.nan_to_num(feat, nan=0.0, posinf=1e10, neginf=-1e10)  # Replace NaN/Inf with 0 or large values
        
        # Append the class label to the corresponding fold's label list
        y_folds[fold - 1].append(df.loc[df['slice_file_name'] == file.name, 'class'].values[0])
        
        # Append the features to the corresponding fold's feature list
        x_folds[fold - 1].append(feat)

    # Initialize the scaler
    scaler = StandardScaler()
    
    # Apply fit_transform to scale the features for each fold
    x_folds = [scaler.fit_transform(np.array(np.vstack(features))) for features in x_folds]
    
    # Return each fold's features and labels as separate variables (x1, y1, ..., x10, y10)
    x1, y1 = x_folds[0], y_folds[0]
    x2, y2 = x_folds[1], y_folds[1]
    x3, y3 = x_folds[2], y_folds[2]
    x4, y4 = x_folds[3], y_folds[3]
    x5, y5 = x_folds[4], y_folds[4]
    x6, y6 = x_folds[5], y_folds[5]
    x7, y7 = x_folds[6], y_folds[6]
    x8, y8 = x_folds[7], y_folds[7]
    x9, y9 = x_folds[8], y_folds[8]
    x10, y10 = x_folds[9], y_folds[9]
    
    return x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10


def createXtrYtr(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, trFolds, testFolds):
    """ 
    This function creates the training and testing datasets based on the provided folds.
    It concatenates the features (X) and labels (Y) from the specified training folds and test folds.

    Parameters:
    -----------
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10 : numpy arrays
        These are the features (x) and labels (y) for each of the 10 folds.
        Each xi is a 2D numpy array of shape (n_samples, n_features), and each yi is a 1D array of shape (n_samples,).
    
    trFolds : list
        A list containing the fold numbers to be used for training (e.g., [1, 2, 3] for folds 1, 2, and 3).
        
    testFolds : list
        A list containing the fold numbers to be used for testing (e.g., [4] for fold 4).
        
    Returns:
    --------
    X_train, Y_train, X_test, Y_test : numpy arrays
        - X_train is the stacked feature matrix for the training folds.
        - Y_train is the stacked label vector for the training folds.
        - X_test is the stacked feature matrix for the testing folds.
        - Y_test is the stacked label vector for the testing folds.
    """
    
    # Create lists to collect the features and labels for training and testing sets
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    # Loop through the training folds and collect corresponding features and labels
    for fold in trFolds:
        # Access the features and labels for each fold
        X_train.append(globals()[f'x{fold}'])  # Using globals() to access x1, x2, ..., x10
        Y_train.append(globals()[f'y{fold}'])  # Using globals() to access y1, y2, ..., y10
    
    # Loop through the test folds and collect corresponding features and labels
    for fold in testFolds:
        # Access the features and labels for each fold
        X_test.append(globals()[f'x{fold}'])  # Using globals() to access x1, x2, ..., x10
        Y_test.append(globals()[f'y{fold}'])  # Using globals() to access y1, y2, ..., y10

    # Stack the training data (features and labels) and testing data (features and labels)
    X_train = np.vstack(X_train)  # Stack the features for training
    Y_train = np.concatenate(Y_train)  # Concatenate the labels for training
    
    X_test = np.vstack(X_test)  # Stack the features for testing
    Y_test = np.concatenate(Y_test)  # Concatenate the labels for testing

    return X_train, Y_train, X_test, Y_test


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