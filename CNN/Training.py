import tensorflow as tf
from tensorflow.keras import models, layers, datasets
import Tensor_images
import numpy as np
import matplotlib as plt

'''
Run basic CNN on the tensor representation of spectrogram data

'''

genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
# Number of songs in each genre
training_set_size = 70
testing_set_size = 30
# Number of genres
num_genres = 10
batch_size = training_set_size * num_genres
# Height and width of images
img_height = 368
img_width = 495
''' 
tensor shape : (700, 495, 368, 3)
'''


'''
LOAD DATA FROM OBJECT FILE

'''


training_labels = np.zeros((training_set_size, num_genres))
testing_labels = np.zeros((testing_set_size, num_genres))

for row in range(num_genres):
    training_labels[row, :] = row
    testing_labels[row, :] = row
print(training_labels)
print(testing_labels)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_height, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

training_tensor = tf.convert_to_tensor(training_data)
testing_tensor = tf.convert_to_tensor(testing_data)
training_label_tensor = tf.convert_to_tensor(training_labels)
testing_label_tensor = tf.convert_to_tensor(testing_labels)

history = model.fit(training_data, training_labels, epochs=10,
                    validation_data=(testing_data, testing_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(testing_data,  testing_labels, verbose=2)
