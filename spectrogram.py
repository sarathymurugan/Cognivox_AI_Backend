import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio, sr = librosa.load("dataset/mysample1.wav", sr=16000)

D = librosa.stft(audio)

DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(12,5))

librosa.display.specshow(
    DB,
    sr=sr,
    x_axis='time',
    y_axis='hz',
    cmap='magma'
)

plt.colorbar(label='dB')
plt.title("Spectrogram")

plt.tight_layout()
plt.show()