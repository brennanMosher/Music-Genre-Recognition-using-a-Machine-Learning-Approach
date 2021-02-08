import Training

# Training/Testing directory locations
training_loc = r'/Music-Genre-Recognition-using-a-Machine-Learning-Approach/Dataset/Spectrogram Dataset Training/'
testing_loc = r'/Music-Genre-Recognition-using-a-Machine-Learning-Approach/Dataset/Spectrogram Dataset Testing/'
splice_train_loc = r'/Music-Genre-Recognition-using-a-Machine-Learning-Approach\Dataset\Spliced Spectrogram\Training/'
splice_testing_loc = r'/Music-Genre-Recognition-using-a-Machine-Learning-Approach\Dataset\Spliced Spectrogram\Testing/'

# TODO get epochs for no overfitting
#EPOCHS
epochs = 30

# UNCHANGED PARAMETERS
# Number of genres
num_genres = 10
# Height and width of images
img_height = 370
img_width = 497

# Default parameters
train = training_loc
test = testing_loc
dataset_size = 700
# Batch size for dataset creation and training
# High batch_size has high computational complexity
batch_size = 100
# Size of Conv2D filters
filter1 = 32
filter2 = 64
# Size of Conv2D kernel
kernel_size = (3, 3)
# Percentage of dataset used for validation
validation_split = 0.3
# Size of strides
strides = (2, 2)

txt_name = 'train.txt'

Training.training(train, test, num_genres, dataset_size, batch_size, img_height, img_width,
					  filter1, filter2, kernel_size, epochs, validation_split, strides, txt_name)

# TODO Save model for use later