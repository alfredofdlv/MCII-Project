import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("Librosa version:", librosa.__version__)
print("NumPy version:", np.__version__)
print("Matplotlib version:", plt.__version__)
print("TensorFlow version:", tf.__version__)

#load example file
audio_path = 'c:/Users/leonor/Desktop/ACII...folders/MCII-project/UrbanSound8K/100032-3-0-0.wav'
y, sr = librosa.load(audio_path, sr=None)

#dispplay the waveform
plt.figure(figsize = (10,4))
librosa.display.waveshow(y,sr=sr)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

#convert audio to mel-spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)

#display mel-spectrogram
plt.figure(figsize=(10,4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-Spectrogram')
plt.slabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()