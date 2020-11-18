import tensorflow as tf
from tensorflow.keras import models, layers, datasets
import matplotlib as plt
import Tensor_images
import Training_help as TH

'''
Run basic CNN on the tensor representation of spectrogram data

RUNNING REALLY SLOW RIGHT NOW BECAUSE THE LAYERS ARE AT HIGH VALUES 
See comments above layers for reasoning
It still works it's just heavy in computation. If we can run with these high values that would be great
Use smaller values for testing
I'll add more intuitive ways to change these values later with defaults

tensor shape : (700, 497, 370, 3)

'''

def main():
	# Training/Testing directory locations
	training_loc = r'/Music-Genre-Recognition-using-a-Machine-Learning-Approach/Dataset/Spectrogram Dataset Training/'
	testing_loc = r'/Music-Genre-Recognition-using-a-Machine-Learning-Approach/Dataset/Spectrogram Dataset Testing/'
	# Genre labels to print out testing results in human readable format
	genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
	# Number of songs in each genre
	training_set_size = 70
	testing_set_size = 30
	# Number of genres
	num_genres = 10
	# Batch size for dataset creation and training
	# High batch_size has high computational complexity
	dataset_size = 699
	batch_size = 20
	# Height and width of images
	img_height = 370
	img_width = 497

	training_path, testing_path = TH.get_data_path(training_loc, testing_loc)

	# Return dataset objects from tensor creation function
	training_dataset = Tensor_images.dataset_to_tensors(training_path, batch_size, dataset_size)
	testing_dataset = Tensor_images.dataset_to_tensors(testing_path, 300, 300)

	# TODO Set layers to new values based on research
	model = models.Sequential()
	'''Filters: Dimensionality of output space (# of filters applied)
	Kernel_size: (height, width) of the 2D convolution window
	
	Kernel size should be small since what separates spectrograms is small differences
	Based on this link:
	https://www.researchgate.net/post/How-do-we-choose-the-filters-for-the-convolutional-layer-of-a-Convolution-Neural-Network-CNN
	Could either use 3x3 or 5x5 if we want think bigger differences between genres occur
	
	The greater the number of filters the higher test accuracy
	As the number increases the complexity increases
	Want to make it as large as possible without being unable to handle the code
	From: The Impact of Filter Size and Number of Filters on Classification Accuracy in CNN (2020)
	'''
	model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
	'''
	A problem with the output feature maps is that they are sensitive to the location of the features in the input. 
	One approach to address this sensitivity is to down sample the feature maps.
	
	This line does the pooling
	
	Can either use 2 or 3. Maybe do some testing later to find which performs better
	'''
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))

	'''
	Maybe think about adding more layers to increase accuracy
	'''

	model.summary()

	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(10))

	model.summary()

	model.compile(optimizer='adam',
				  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				  metrics=['accuracy'])

	# TODO: Need to test number of epochs to find when overfitting occurs
	history = model.fit(training_dataset, epochs=20, validation_data=testing_dataset)
	# TODO: Need to implement cross validation

	# Plotting not working since matplotlib doesn't have plot function. Not super familiar with matplotlib so maybe I'm doing something wrong
	plt.plot(history.history['accuracy'], label='accuracy')
	plt.plot(history.history['val_accuracy'], label='val_accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0.5, 1])
	plt.legend(loc='lower right')

	test_loss, test_acc = model.evaluate(testing_dataset, verbose=2)

if __name__ == '__main__':
	main()
