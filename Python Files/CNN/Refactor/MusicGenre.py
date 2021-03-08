import tensorflow as tf
from tensorflow.keras import models, layers
import sys


class MusicGenre:
	def __init__(self):

		self.model = models.Sequential()
		self.model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu',
								input_shape=(370, 497, 3)))
		self.model.add(layers.MaxPooling2D((2, 2)))
		self.model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
		self.model.add(layers.MaxPooling2D((2, 2)))
		self.model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu'))

		self.model.summary()

		self.model.add(layers.Flatten())
		self.model.add(layers.Dense(64, activation='relu'))
		self.model.add(layers.Dense(10))

		self.model.summary()

		self.model.compile(optimizer='adam',
					  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
					  metrics=['accuracy'])

	def train(self, training_dataset, validation_dataset, epochs):

		self.model.fit(training_dataset, epochs=epochs, validation_data=validation_dataset)
		self.model.save('model_v1')

		return self.model

	def test(self, testing_dataset):

		# Return (test_loss, test_acc)
		return self.model.evaluate(testing_dataset, verbose=1)

