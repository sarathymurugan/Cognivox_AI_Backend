import librosa
import librosa.display
import matplotlib.pyplot as plt

audio, sr = librosa.load("dataset/mysample1.wav", sr=16000)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=13
)

print("MFCC Shape:", mfcc.shape)

plt.figure(figsize=(12, 5))

librosa.display.specshow(
    mfcc,
    x_axis="time",
    cmap="viridis"
)

plt.colorbar(label="Coefficient Value")
plt.title("MFCC Features")
plt.tight_layout()

plt.show()