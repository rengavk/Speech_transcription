import librosa
import matplotlib.pyplot as plt

audio = "C:/my folder/Speechtranscription/data/raw_audio/C01_1002.mp3"

y, sr = librosa.load(audio, sr=None)

plt.figure(figsize=(15,4))
plt.plot(y)
plt.title("Original Audio Waveform")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.savefig("C:/my folder/Speechtranscription/output//figures/waveform.png", dpi=300)

plt.show()