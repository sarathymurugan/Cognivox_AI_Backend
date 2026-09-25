import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import os

audio, sample_rate = librosa.load(
    "dataset/mysample1.wav",
    sr=None
)

print(audio)
print(len(audio))
print("Shape :", audio.shape)
print(sample_rate)

#/stereo to mono audio conversion/
# audio, sr = librosa.load(
#     "dataset/sample.wav",
#     sr=16000,
#     mono=True
# )
# print("New sample rate:", sr)
# print("New shape:", audio.shape)

# audio trimming
trimmed_audio, index = librosa.effects.trim(audio)

print("Original samples :", len(audio))
print("Trimmed samples :", len(trimmed_audio))

base_path = "dataset/trimmed_audio/trimmed.wav"

if not os.path.exists(base_path):
    output_path = base_path
else:
    counter = 1
    while True:
        output_path = f"dataset/trimmed_audio/trimmed{counter}.wav"
        if not os.path.exists(output_path):
            break
        counter += 1

sf.write(
    output_path,
    trimmed_audio,
    sample_rate
)

plt.figure(figsize=(14,4))
plt.subplot(1,2,1)
plt.plot(audio)
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.title("Original")
plt.subplot(1,2,2)
plt.plot(trimmed_audio)
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.title("Trimmed")
plt.tight_layout()
plt.show()

