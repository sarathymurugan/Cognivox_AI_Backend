import librosa
import matplotlib.pyplot as plt
import numpy as np

audio, sr = librosa.load("dataset/mysample1.wav", sr=16000)

pitch = librosa.yin(
    audio,
    fmin=75,
    fmax=300,
    sr=sr
)

times = librosa.times_like(pitch, sr=sr)

plt.figure(figsize=(12,4))
plt.plot(times, pitch)

plt.title("Pitch (Fundamental Frequency)")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.grid(True)

plt.show()

valid_pitch = pitch[~np.isnan(pitch)]

print("Mean Pitch:", np.mean(valid_pitch))
print("Min Pitch:", np.min(valid_pitch))
print("Max Pitch:", np.max(valid_pitch))
print("Std Dev:", np.std(valid_pitch))