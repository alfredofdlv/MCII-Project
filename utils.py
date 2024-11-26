import librosa
import numpy as np 
import pandas as pd 
import random
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


def get_zerocr(direction)  :
    # Cargar el archivo de audio
    y, sr = librosa.load(direction, sr = 22050)
    # # Parámetros de extracción
    frame_length = int(sr * 0.0232)  # Ventana de 23.2 ms en muestras
    hop_length = frame_length // 2   # 50% de superposición
    # mel_spectrogram = librosa.feature.melspectrogram(
    #     y=y, sr=sr, n_mels=40, hop_length=hop_length, n_fft=frame_length, fmax=sr // 2
    # ).T
    # mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # Calcular características
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length).T
    rmse = librosa.feature.rms(y=y, hop_length=hop_length).T
    # spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length).T
    # spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length).T
    # spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length).T
    # spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length).T
    # mel_spectrogram = librosa.feature.melspectrogram(
    #     y=y, sr=sr, n_mels=40, hop_length=hop_length, n_fft=frame_length, fmax=sr // 2
    # ).T
    return np.concatenate([zcr,rmse], axis=1)

def repeat_sound(numpy_list_concatenation, aimed_duration=4, sr=22050, hop_length=512):
    # Duración en segundos de cada ventana
    frame_duration = hop_length / sr
    # Duración total del array original
    original_duration = frame_duration * numpy_list_concatenation.shape[0]
    # Número de repeticiones necesarias para alcanzar la duración deseada
    repetitions = int(np.ceil(aimed_duration / original_duration))
    
    # Repetir las características para alcanzar la duración deseada
    repeated_array = np.tile(numpy_list_concatenation, (repetitions, 1))
    
    # Recortar al tamaño exacto
    required_frames = int(aimed_duration / frame_duration)
    final_array = repeated_array[:required_frames]
    
    return final_array

def extract_data_and_folds(audio_directory, metadata_df, aimed_duration=4):
    """
    Extrae características, etiquetas y el fold de cada archivo de audio.

    Parameters:
    audio_directory (str or Path): Directorio que contiene los archivos de audio.
    metadata_df (DataFrame): DataFrame con la metadata de los audios ('slice_file_name', 'fold', 'classID').

    Returns:
    tuple: (features_list, labels_list, folds_list), donde:
        - features_list: Lista de características extraídas de cada archivo.
        - labels_list: Lista de etiquetas correspondientes.
        - folds_list: Lista de folds a los que pertenece cada archivo.
    """
    audio_directory = Path(audio_directory)
    features_list, labels_list, folds_list = [], [], []

    for file in audio_directory.rglob("*.wav"):
        file_name = file.name
        
        if file_name not in metadata_df['slice_file_name'].values:
            continue  # Ignorar archivos sin metadata

        # Obtener el fold, la etiqueta y las características
        fold = metadata_df.loc[metadata_df['slice_file_name'] == file_name, 'fold'].iloc[0]
        label = metadata_df.loc[metadata_df['slice_file_name'] == file_name, 'classID'].iloc[0]
        
        # Extraer las características
        features = repeat_sound(get_zerocr(file), aimed_duration=aimed_duration)  # Repetir y ajustar duración
        
        features_list.append(features)
        labels_list.append(label)
        folds_list.append(fold)
    
    return features_list, labels_list, folds_list

def prepare_datasets(features_list, labels_list, folds_list, train_folds, test_folds):
    """
    Crea conjuntos de datos de entrenamiento y prueba según los folds especificados.

    Parameters:
    features_list (list): Lista de características extraídas.
    labels_list (list): Lista de etiquetas correspondientes.
    folds_list (list): Lista de folds asociados a cada archivo.
    train_folds (list): Lista de números de folds a usar para entrenamiento.
    test_folds (list): Lista de números de folds a usar para prueba.

    Returns:
    tuple: (X_train, X_test, y_train, y_test) donde:
        - X_train, X_test: Listas escaladas de características para entrenamiento y prueba.
        - y_train, y_test: Listas de etiquetas para entrenamiento y prueba.
    """
    # Dividir en conjuntos de entrenamiento y prueba
    train_features, test_features = [], []
    y_train, y_test = [], []

    for features, label, fold in zip(features_list, labels_list, folds_list):
        if fold in train_folds:
            train_features.append(features)
            y_train.append(label)
        elif fold in test_folds:
            test_features.append(features)
            y_test.append(label)

    # Asegurarse de que todas las características tienen la misma forma
    train_features_combined = np.vstack(train_features)
    scaler = StandardScaler()
    scaler.fit(train_features_combined)

    # Escalar características
    X_train = [scaler.transform(features) for features in train_features]
    X_test = [scaler.transform(features) for features in test_features]

    return X_train, X_test, y_train, y_test

