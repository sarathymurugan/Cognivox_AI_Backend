import librosa
import matplotlib.pyplot as plt

# import audio from file
audio, sample_rate = librosa.load(
    "dataset/sample.wav",
    sr=16000
)

print(audio)
print(len(audio))
print("Shape :", audio.shape)
print(sample_rate)

# picture
plt.figure(figsize=(12,4))
plt.plot(audio)
plt.title("Speech Waveform")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

