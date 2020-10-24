from PIL import Image
import os
import numpy as np

'''
Converts the spectogram dataset to a matrix representation

Matrix is of size 10x70
 - Rows are the number of genres
 - Columns are the song of each genre (index)
 
Each element of the matrix holds a 3D Matrix of the spectrogram in RGB representation
This interior matrix is of the form 495x368 (width by height). This is the size of the cropped spectrograms

Each element of this interior matrix has 3 values (RGB) (int32)
'''

def data_to_matrix(training_loc, genre_size, num_genres):
    # Width is number of songs
    # Height is number of genres
    # Each element of the matrix holds a matrix value representing the image
    dataset_mat = np.zeros((num_genres, genre_size), dtype=object)
    print(dataset_mat.shape)

    # Height of the matrix dataset_mat
    genre_index = 0
    for genre in os.listdir(training_loc):
        # Index in dataset_mat of the spectrogram (width)
        img_index = 0
        for image in os.listdir(training_loc + genre):
            print(image)
            # Open spectrogram as Pillow image
            img_loc = training_loc + genre + '/' + image
            im = Image.open(img_loc)

            # Get bounds of image
            size = im.size
            x_size = size[0]
            y_size = size[1]


            # Init matrix to hold pixel map of single image
            # Reset every iteration (each image)
            pixel_mat = np.zeros((x_size,y_size), dtype=(int , 3))
            # Maybe intuitive way with pillow to loop through image
            for x in range(x_size-1):
                for y in range(y_size-1):
                    # Temp matrix to hold pixel map of image
                    pixel_mat[x, y] = im.getpixel((x,y))

            # Remove extreme values (That have value RGB(0,0,0)_
            pixel_mat = np.delete(pixel_mat, (x_size-1), axis=0)
            pixel_mat = np.delete(pixel_mat, (y_size-1), axis=1)
            pixel_mat = np.delete(pixel_mat, (0), axis=0)
            pixel_mat = np.delete(pixel_mat, (0), axis=1)

            # Update matrix size
            x_size = x_size - 2
            y_size = y_size - 2


            dataset_mat[genre_index, img_index] = pixel_mat
            img_index = img_index + 1
        genre_index = genre_index + 1
    print(dataset_mat)
    return dataset_mat