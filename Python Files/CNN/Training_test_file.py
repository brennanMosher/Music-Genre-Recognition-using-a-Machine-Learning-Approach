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
batch_size = 1
# Size of Conv2D filters
filter1 = 32
filter2 = 64
# Size of Conv2D kernel
kernel_size = (3, 3)
# Percentage of dataset used for validation
validation_split = 0.3
# Size of strides
strides = (2, 2)


# Parameter ranges
batch_size_test = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 700]
#filter1_test = [32, 64]
#filter2_test = [64, 128]
kernel_size_test = [(3, 3), (5, 5)]
validation_split_test = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
strides_test = [(1, 1), (2, 2), (4, 4)]

'''
# Batch size testing
for x in batch_size_test:
	batch_size = x
	txt_name = 'batch_size' + str(x) + '.txt'
	Training.training(train, test, num_genres, dataset_size, batch_size, img_height, img_width,
				  filter1, filter2, kernel_size, epochs, validation_split, strides, txt_name)
batch_size = 1'''

# Kernel testing
for x in kernel_size_test:
	kernel_size = x
	txt_name = 'kernel_size' + str(x) + '.txt'
	Training.training(train, test, num_genres, dataset_size, batch_size, img_height, img_width,
					  filter1, filter2, kernel_size, epochs, validation_split, strides, txt_name)
kernel_size = (3, 3)

# Validation split testing
for x in validation_split_test:
	validation_split = x
	txt_name = 'validation_split' + str(x) + '.txt'
	Training.training(train, test, num_genres, dataset_size, batch_size, img_height, img_width,
					  filter1, filter2, kernel_size, epochs, validation_split, strides, txt_name)
validation_split = 0.3

# Strides testing
for x in strides_test:
	strides = x
	txt_name = 'strides' + str(x) + '.txt'
	Training.training(train, test, num_genres, dataset_size, batch_size, img_height, img_width,
					  filter1, filter2, kernel_size, epochs, validation_split, strides, txt_name)
strides = (2, 2)


train = splice_train_loc
test = splice_testing_loc
dataset_size = 7000
txt_name = 'splice.txt'
Training.training(train, test, num_genres, dataset_size, batch_size, img_height, img_width,
				  filter1, filter2, kernel_size, epochs, validation_split, strides, txt_name)