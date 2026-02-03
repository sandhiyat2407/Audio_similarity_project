import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.fft import fft

# Load audio files (force same sample rate)
audio1, sr = librosa.load("audio1.mp3", sr=22050)
audio2, _ = librosa.load("audio2.mp3", sr=22050)

# Make both audios same length
min_len = min(len(audio1), len(audio2))
audio1 = audio1[:min_len]
audio2 = audio2[:min_len]

# =======================
# SIMILARITY CALCULATION
# =======================

# Extract MFCC features
mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr, n_mfcc=13)
mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr, n_mfcc=13)

# Take mean of MFCCs
mfcc1_mean = np.mean(mfcc1, axis=1)
mfcc2_mean = np.mean(mfcc2, axis=1)

# Cosine similarity
similarity = cosine_similarity(
    mfcc1_mean.reshape(1, -1),
    mfcc2_mean.reshape(1, -1)
)[0][0]

similarity_percentage = similarity * 100

print(f"\n🎵 Audio Similarity Percentage: {similarity_percentage:.2f}%")

# =======================
# FREQUENCY DOMAIN GRAPH
# =======================

fft_audio1 = np.abs(fft(audio1))[:min_len // 2]
fft_audio2 = np.abs(fft(audio2))[:min_len // 2]

freqs = np.linspace(0, sr / 2, min_len // 2)

plt.figure(figsize=(10, 5))
plt.plot(freqs, fft_audio1, label="Audio 1", alpha=0.7)
plt.plot(freqs, fft_audio2, label="Audio 2", alpha=0.7)

plt.title("Frequency Domain Comparison of Two Audio Signals")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()