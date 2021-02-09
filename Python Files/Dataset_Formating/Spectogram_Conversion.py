import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from PIL import Image
<<<<<<< HEAD
import os
from pydub import AudioSegment
import pydub
import Stereo_Mono_Conversion

=======
>>>>>>> parent of 25ae312 (Spectrogram for new dataset (Not working with new data))

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

<<<<<<< HEAD
    return frequencies, times, spectrogram

def main():
    path = r'C:\Users\brenn\Documents\GitHub\Music-Genre-Recognition-using-a-Machine-Learning-Approach\Dataset\FMA dataset\Sorted/'

    '''
    for folder in os.listdir(path):
        loc = path + folder + '/'
        for file in os.listdir(loc):
            if file.endswith('.mp3'):
                src = loc+file
                print(src)
                dst = loc+file[:-4]+'.wav'
                sound = AudioSegment.from_file(src)
                sound.export(dst, format='wav')
    '''

    for folder in os.listdir(path):
        loc = path + folder + '/'
        for file in os.listdir(loc):
            if file.endswith('.wav'):
                print(loc)
                print(file)
                temp = Stereo_Mono_Conversion.Stereo_Mono_Conversion(loc, file)
                frequencies, times, spectrogram = spectrogram_conversion(temp[:-15], temp[:-15], 'mono_042129.wav', True)


if __name__ == '__main__':
    main()
=======
    return frequencies, times, spectrogram
>>>>>>> parent of 25ae312 (Spectrogram for new dataset (Not working with new data))
