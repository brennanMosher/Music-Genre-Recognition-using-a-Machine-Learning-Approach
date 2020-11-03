import tensorflow as tf
from tensorflow.keras import models, layers, datasets
import Tensor_images
import numpy as np
import matplotlib as plt
import Tensor_images

'''
Run basic CNN on the tensor representation of spectrogram data
'''

# NEED TO CHANGE THESE TO PERSONAL DIRECTORY
# IF SOMEONE CAN GENERALIZE THAT WOULD BE GREAT
training_loc = r'C:\Users\brenn\Documents\GitHub\Music-Genre-Recognition-using-a-Machine-Learning-Approach\Dataset' \
               r'\Spectrogram Dataset Training/'
testing_loc = r'C:\Users\brenn\Documents\GitHub\Music-Genre-Recognition-using-a-Machine-Learning-Approach\Dataset' \
              r'\Spectrogram Dataset Testing/'

genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
# Number of songs in each genre
training_set_size = 70
testing_set_size = 30
# Number of genres
num_genres = 10
batch_size = 699
# Height and width of images
img_height = 370
img_width = 497
''' 
tensor shape : (700, 497, 370, 3)
'''

# Return dataset objects from tensor creation function
training_dataset = Tensor_images.dataset_to_tensors(training_loc, 20, batch_size)
testing_dataset = Tensor_images.dataset_to_tensors(testing_loc, 300, 300)
print(training_dataset)


# Create actual CNN model
# Not completely sure on how values of this are decided
# Will need to do some extra research to know what to set values to
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
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

# TRAIN THE MODEL
# Increase number of epochs
history = model.fit(training_dataset, epochs=10, validation_data=testing_dataset)
# Still need to implement cross validation

# Plotting not working since matplotlib doesn't have plot function. Not super familiar with matplotlib so maybe I'm doing something wrong
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(testing_dataset, verbose=2)
