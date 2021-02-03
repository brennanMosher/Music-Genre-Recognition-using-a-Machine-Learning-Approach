import tensorflow as tf
from tensorflow.keras import models, layers, datasets
import matplotlib.pyplot as plt
import Tensor_images
import Training_help as TH
from sklearn.model_selection import train_test_split

# Training/Testing directory locations
training_loc = r'/Music-Genre-Recognition-using-a-Machine-Learning-Approach/Dataset/Spectrogram Dataset Training/'
testing_loc = r'/Music-Genre-Recognition-using-a-Machine-Learning-Approach/Dataset/Spectrogram Dataset Testing/'
splice_train_loc = r'/Music-Genre-Recognition-using-a-Machine-Learning-Approach\Dataset\Spliced Spectrogram\Training/'
splice_testing_loc = r'/Music-Genre-Recognition-using-a-Machine-Learning-Approach\Dataset\Spliced Spectrogram\Testing/'

train = splice_train_loc
test = splice_testing_loc

training_path, testing_path = TH.get_data_path(train, test)

testing_dataset = Tensor_images.dataset_to_tensors(testing_path, 300, 300)

model = tf.keras.models.load_model('model_v1')

# Check its architecture
model.summary()

print('testing')

test_loss, test_acc = model.evaluate(testing_dataset, verbose=1)

print(test_loss)
print(test_acc)
