import librosa
import numpy as np 
import pandas as pd 
import random
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

def preprocess_data(X_train, X_test, y_train, y_test):
    """
    Preprocesa los datos de entrada y las etiquetas para su uso en modelos de aprendizaje profundo.
    
    Parámetros:
        X_train (list or np.ndarray): Secuencias de entrenamiento.
        X_test (list or np.ndarray): Secuencias de prueba.
        y_train (list or np.ndarray): Etiquetas de entrenamiento.
        y_test (list or np.ndarray): Etiquetas de prueba.
    
    Devuelve:
        tuple: (X_train_padded, X_test_padded, y_train_one_hot, y_test_one_hot, label_encoder, max_timesteps)
    """
    # Calcular las longitudes de las secuencias en el conjunto de entrenamiento
    sequence_lengths = [len(seq) for seq in X_train]
    
    # Crear el codificador de etiquetas
    label_encoder = LabelEncoder()
    
    # Ajustar y transformar las etiquetas de entrenamiento
    y_train_numeric = label_encoder.fit_transform(y_train)
    
    # Transformar las etiquetas de prueba usando el mismo codificador
    y_test_numeric = label_encoder.transform(y_test)
    
    # Imprimir las clases asignadas
    print("Clases:", label_encoder.classes_)
    
    # Definir la longitud máxima deseada (máximo de las longitudes de las secuencias)
    max_timesteps = max(sequence_lengths)  # Ajustar si es necesario
    
    # Aplicar padding a las secuencias
    X_train_padded = pad_sequences(X_train, maxlen = max_timesteps, padding='post', dtype='float32')
    X_test_padded = pad_sequences(X_test, maxlen = max_timesteps, padding='post', dtype='float32')
    
    # Verificar las dimensiones de los datos después del padding
    print("Forma de X_train después de padding:", X_train_padded.shape)
    print("Forma de X_test después de padding:", X_test_padded.shape)
    
    # Determinar el número de clases únicas
    num_classes = len(np.unique(y_train))
    
    # Convertir las etiquetas a one-hot encoding
    y_train_one_hot = to_categorical(y_train_numeric, num_classes=num_classes)
    y_test_one_hot = to_categorical(y_test_numeric, num_classes=num_classes)
    
    return X_train_padded, X_test_padded, y_train_one_hot, y_test_one_hot, label_encoder, max_timesteps

def pad_features_dynamic(features, padding_value=0, target_length=250):
    """
    Realiza padding o truncado para ajustar vectores a la longitud del más largo.

    Parameters:
    -----------
    features : list of np.ndarray
        Lista de vectores de características con diferentes longitudes.
    padding_value : float, optional
        Valor para rellenar las características si son más cortas que el objetivo (por defecto, 0).
    target_length : int
        Longitud objetivo a la que se ajustarán todas las características.

    Returns:
    --------
    np.ndarray
        Array de NumPy donde todas las características tienen la longitud del vector más largo.
    """
    padded_features = []
    for feat in features:
        feat = np.array(feat)  # Convertir a array por seguridad
        if len(feat) < target_length:
            # Rellenar con el valor especificado si la longitud es menor
            feat = np.pad(feat, (0, target_length - len(feat)), constant_values=padding_value)
        else:
            # Truncar si la longitud es mayor
            feat = feat[:target_length]
        padded_features.append(feat)
    
    return np.array(padded_features)

def get_features(direction,target_length = None):
    """
    Extrae características de un archivo de audio y las ajusta dinámicamente al tamaño máximo.
    
    Parameters:
    -----------
    direction : str
        Ruta del archivo de audio.

    Returns:
    --------
    np.ndarray
        Matriz de características ajustada dinámicamente a la longitud del vector más largo.
    """
    # Cargar el archivo de audio
    y, sr = librosa.load(direction, sr = 1600)

    # Parámetros de extracción
    frame_length = int(sr * 0.0232)  # Ventana de 23.2 ms en muestras
    hop_length = frame_length // 2   # 50% de superposición

    # Calcular características
    """
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length).T
    rmse = librosa.feature.rms(y=y, hop_length=hop_length).T
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=25, hop_length=hop_length, n_fft=frame_length).T
    """
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length).T
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length).T
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length).T
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length).T
    """
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=40, hop_length=hop_length, n_fft=frame_length, fmax=sr // 2
    ).T
    
    """
    # Ajustar el ancho de la delta
    width = min(9, mfccs.shape[1])
    if width < 3:
        width = 3
    elif width % 2 == 0:
        width -= 1
    
    # Calcular derivadas (delta)
    delta_mfccs = librosa.feature.delta(mfccs, width=width).T
    delta2_mfccs = librosa.feature.delta(mfccs, order=2, width=width).T
    """
    mel_specgram_norm = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std() # Noramalization
    mfcc_norm = (mfccs - mfccs.mean()) / mfccs.std()
    # Ajustar características dinámicamente
    feature_list = [
            mfccs, mel_spectrogram
        ]
    
    # Calcular un target_length común
    #if target_length is None:
    #    target_length = max(len(feat) for feat in feature_list)
    
    # Aplicar padding a todas las características
    #feature_list = [pad_features_dynamic(feat, target_length = target_length) for feat in feature_list]

    # Concatenar todas las características a lo largo del eje 1
    features = np.concatenate(feature_list, axis=1)

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

def create_dataset(audio_directory, metadata_df, train_folds, test_folds):
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
    audio_directory = Path(audio_directory)
    train_features, test_features = [], []
    y_train, y_test = [], []

    #     # Extract features for each audio file
    for file in audio_directory.rglob("*.wav"):
        fold = metadata_df.loc[metadata_df['slice_file_name'] == file.name, 'fold'].iloc[0]
        features = get_features(file)

        if fold in train_folds:
            y_train.append(metadata_df.loc[metadata_df['slice_file_name'] == file.name, 'class'].iloc[0])
            train_features.append(features)
        
        else:
            y_test.append(metadata_df.loc[metadata_df['slice_file_name'] == file.name, 'class'].iloc[0])
            test_features.append(features)

    # Scaling features
    train_features_combined = np.vstack(train_features)
    scaler = StandardScaler()
    scaler.fit(train_features_combined)

    X_train = [scaler.transform(features) for features in train_features]
    X_test = [scaler.transform(features) for features in test_features]

    return X_train, X_test, y_train, y_test
"""
def create_folds(directory_audio, df):
    
    Divide el conjunto de datos en 10 pliegues basados en la columna 'fold' del DataFrame.
    Cada pliegue contiene características (x) y etiquetas (y).
    
    directory_audio = Path(directory_audio)
    
    # Inicializar listas para almacenar características (x) y etiquetas (y) para cada pliegue
    x_i(globals) = [] for i in range(1,11)
    y_i(globals) = [] for i in range(1,11)
    
    for file in directory_audio.rglob("*.wav"):
        fold = df.loc[df['slice_file_name'] == file.name, 'fold'].iloc[0]
        feat = get_features(file)

        if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
            feat = np.nan_to_num(feat, nan=0.0, posinf=1e10, neginf=-1e10)
        
        y_globals()[fold].append(df.loc[df['slice_file_name'] == file.name, 'class'].values[0])
        x_globals()[fold].append(feat)
    
    # Escalamiento
    scaler = StandardScaler()
    x_i(globals())

    # Convertir las listas en arreglos numpy bidimensionales
    return [(np.array(x_folds[i]), np.array(y_folds[i])) for i in range(10)]
"""

def createXtrYtr(folds, trFolds, testFolds):
    """
    Genera conjuntos de entrenamiento y prueba en base a los pliegues indicados.
    """
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    
    for fold in trFolds:
        X_train.append(folds[fold - 1][0])  # Agregar características del pliegue
        Y_train.append(folds[fold - 1][1])  # Agregar etiquetas del pliegue
    
    for fold in testFolds:
        X_test.append(folds[fold - 1][0])
        Y_test.append(folds[fold - 1][1])
    
    # Concatenar los pliegues
    X_train = np.vstack(X_train)
    Y_train = np.concatenate(Y_train)
    X_test = np.vstack(X_test)
    Y_test = np.concatenate(Y_test)
    
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