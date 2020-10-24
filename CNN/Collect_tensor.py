import numpy as np
import Tensor_images
from numpy import savetxt


'''
File to run converting to matrix for training and testing data

Save the matrix to .csv file using numpy savetxt

Matrix construction takes a long time so reading from file is much quicker
 - Only run this once
'''


# Change these to personal github directory on device
training_loc = r'C:\Users\brenn\Documents\GitHub\Music-Genre-Recognition-using-a-Machine-Learning-Approach\Dataset' \
               r'\Spectrogram Dataset Training/'
testing_loc = r'C:\Users\brenn\Documents\GitHub\Music-Genre-Recognition-using-a-Machine-Learning-Approach\Dataset' \
               r'\Spectrogram Dataset Testing/'

# Number of songs in each genre
training_set_size = 70
testing_set_size = 30
# Number of genres
num_genres = 10

training_data = Tensor_images.data_to_matrix(training_loc, training_set_size, num_genres)
testing_data = Tensor_images.data_to_matrix(testing_loc, testing_set_size, num_genres)

savetxt('training_data.csv', training_data, delimiter=',')
savetxt('testing_data.csv', testing_data, delimiter=',')