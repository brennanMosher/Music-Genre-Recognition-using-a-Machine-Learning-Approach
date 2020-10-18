import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile


def spectrogram_conversion(path, output_path, audio_file, plot=True):
    print(path)
    print(audio_file)

    # Get Spectrogram data from .wav file
    sample_rate, samples = wavfile.read(path)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    # Display and save the spectrogram
    if plot is True:
        plt.pcolormesh(times, frequencies, 10*np.log10(spectrogram), shading='gouraud')
        np.log(spectrogram)
        # Set output path for images
        spectogram_dest = output_path + '/' + audio_file[:-4]
        print(spectogram_dest)
        plt.savefig(spectogram_dest + '.jpg')

        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        plt.close('all')

    return frequencies, times, spectrogram