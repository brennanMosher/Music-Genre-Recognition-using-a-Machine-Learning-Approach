import tensorflow as tf
from tensorflow.keras import models, layers, datasets
import matplotlib.pyplot as plt
import Tensor_images
import Training_help as TH
import sys
from sklearn.model_selection import train_test_split

'''
Run basic CNN on the tensor representation of spectrogram data
tensor shape : (700, 497, 370, 3)
'''

def training(training_loc, testing_loc, num_genres, dataset_size, batch_size, img_height, img_width, filter1, filter2,
			 kernel_size, epochs, validation_split, strides, txt_name):

	# Print to txt file
	sys.stdout = open(txt_name, 'w')

	# Genre labels to print out testing results in human readable format
	genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

	# TODO testing with spliced data
	training_path, testing_path = TH.get_data_path(training_loc, testing_loc)

	# Return dataset objects from tensor creation function
	training_dataset = Tensor_images.dataset_to_tensors(training_path, batch_size, dataset_size)
	testing_dataset = Tensor_images.dataset_to_tensors(testing_path, 300, 300)

	print('dataset_size')
	print(len(training_dataset))

	split = int(len(training_dataset)*validation_split)
	# Skip and take are used to split the training dataset into a training and validation set
	validation_dataset = training_dataset.take(split)
	training_dataset = training_dataset.skip(split)
	print(len(training_dataset))
	print(len(validation_dataset))

	model = models.Sequential()
	# TODO test with different filter values (64, 128, 128)
	# TODO test Kernel size with (5,5)
	# TODO test with different activation functions
	# TODO see if increasing the strides helps with computational complexity
	model.add(layers.Conv2D(filters=filter1, kernel_size=kernel_size, strides=strides, activation='relu', input_shape=(img_height, img_width, 3)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(filters=filter2, kernel_size=kernel_size, strides=strides, activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(filters=filter2, kernel_size=kernel_size, strides=strides, activation='relu'))


	model.summary()

	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(10))

	model.summary()

	model.compile(optimizer='adam',
				  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				  metrics=['accuracy'])

	# TODO: Need to test number of epochs to find when overfitting occurs
	history = model.fit(training_dataset, epochs=epochs, validation_data=validation_dataset)
	# TODO: Need to implement cross validation


	'''
	# Plotting not working since matplotlib doesn't have plot function. Not super familiar with matplotlib so maybe I'm doing something wrong
	plt.plot(history.history['accuracy'], label='accuracy')
	plt.plot(history.history['val_accuracy'], label='val_accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0.5, 1])
	plt.legend(loc='lower right')
	'''

	test_loss, test_acc = model.evaluate(testing_dataset, verbose=1)

	print('testing')
	print(test_loss)
	print(test_acc)
	
	sys.stdout.close()

	return
