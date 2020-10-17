'''
Helper file to sort files into training and testing sets

Never use again since data is sorted
'''

import os

file_dir = r'dataset/genres_original/'

training_threshold = 70
# testing_threshold = 30

# Loop through all the genre folders
for folder in os.listdir(file_dir):
    # Create folders for testing and training data of each genre
    os.mkdir(file_dir + folder + '/' + folder +'_train')
    os.mkdir(file_dir + folder + '/' + folder +'_test')

    genre_dir = file_dir + folder + '/'
    song_count = 0
    # For all audio files in each folder
    for genre_file in os.listdir(genre_dir):
        # If audio file
        if genre_file.endswith('.wav'):
            print(genre_file)
            # Training threshold limits the number of files to be set to training set
            if song_count < training_threshold:
                os.rename(file_dir + folder + '/' + genre_file,
                          file_dir + folder + '/' + folder + '_train/' + genre_file)
            else:
                os.rename(file_dir + folder + '/' + genre_file,
                          file_dir + folder + '/' + folder + '_test/' + genre_file)
            # Update song count
            song_count += 1
