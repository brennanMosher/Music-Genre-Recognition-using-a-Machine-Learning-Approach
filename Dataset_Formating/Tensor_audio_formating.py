import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile as wav

def print_FFT(song_path):

    rate, data = wav.read(song_path)
    fft_out = fft(data)

    plt.plot(data, np.abs(fft_out))
    plt.show()

def convert_audio_tensor(song_path):

    # Convert wav file to Tensor
    audio = tfio.audio.AudioIOTensor(song_path)
    # Splice the audio file
    audio_splice = audio[100:]
    # Remove last dimension
    audio_tensor = tf.squeeze(audio_splice, axis=[-1])

    rate, data = wav.read(song_path)
    max_data = max(data)


    # Prepare the audio file for printing
    # Cast to float value
    tensor = tf.cast(audio_tensor, tf.float32) / max_data

    spectrogram = tfio.experimental.audio.spectrogram(
        tensor, nfft=512, window=512, stride=1024)


# Function to plot the spectrogram from TensorFlow conversion
def plot_spectrogram(spectrogram):
    plt.figure()
    plt.imshow(tf.math.log(spectrogram).numpy())
    plt.show()