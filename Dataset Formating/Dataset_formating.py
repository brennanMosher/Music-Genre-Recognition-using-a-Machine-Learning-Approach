import os
import Spectogram_Conversion as sc
import Stereo_Mono_Conversion as sm


file_dir = r'C:/Users/brenn/Documents/GitHub/Music-Genre-Recognition-using-a-Machine-Learning-Appraoch/Dataset/genres_original/'


for genre in os.listdir(file_dir):
    for folder in os.listdir(file_dir + genre + '/'):
        print(folder)
        # Loop through training set
        folder_dir = file_dir + genre + '/' + folder + '/'

        # Loop through training dataset
        for audio_file in os.listdir(folder_dir):
            print(audio_file)
            # Extra check to ensure working on .wav files
            if audio_file.endswith('.wav'):
                output_path = sm.Stereo_Mono_Conversion(folder_dir, audio_file)
                spectrogram_dir = folder_dir + '/Spectograms/'
                frequencies, times, spectrogram = sc.spectrogram_conversion(output_path, spectrogram_dir, audio_file, plot=True)