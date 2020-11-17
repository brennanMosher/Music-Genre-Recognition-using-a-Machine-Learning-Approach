import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from PIL import Image

'''
Convert .wav file to spectrogram representation 

Should change to input file type to make more general 
    - Should accept mp3, mp4, FLAC, ...
'''


def spectrogram_conversion(path, output_path, audio_file, plot=True):
    # Get Spectrogram data from .wav file
    sample_rate, samples = wavfile.read(path+audio_file)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    # Display and save the spectrogram
    if plot is True:
        norm = [float(i) / max(frequencies) for i in frequencies]

        plt.pcolormesh(times, norm, 10*np.log10(spectrogram), shading='gouraud')
        np.log(spectrogram)
        # Set output path for images
        #spectogram_dest = output_path + '/' + audio_file[:-4]
        spectogram_dest = output_path + audio_file[:-4]
        # print(spectogram_dest)
        plt.savefig(spectogram_dest + '.jpg')

        img = Image.open(spectogram_dest + '.jpg')

        # (left, upper, rigth, lower)
        crop_box = (80, 58, 577, 428)
        img1 = img.crop(crop_box)
        img1.save(spectogram_dest + '.jpg')

        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()
        plt.close('all')

    return frequencies, times, spectrogram