import tensorflow as tf
import Tensor_images
import numpy as np

# Training/Testing directory locations
#training_loc = r'\..\School\Fourth Year\ELEC 490_Electrical Engineering Project\Dataset(Current)\Spectrogram Dataset Training'
#testing_loc = r'\..\School\Fourth Year\ELEC 490_Electrical Engineering Project\Dataset(Current)\Spectrogram Dataset Testing'
#splice_train_loc = r'/Music-Genre-Recognition-using-a-Machine-Learning-Approach\Dataset\Spliced Spectrogram\Training/'
#splice_testing_loc = r'/Music-Genre-Recognition-using-a-Machine-Learning-Approach\Dataset\Spliced Spectrogram\Testing/'

#train = training_loc
#test = testing_loc

#training_path, testing_path = TH.get_data_path(train, test)

training_path = r'C:\Users\brenn\Documents\School\Fourth Year\ELEC 490_Electrical Engineering Project\Dataset(Current)\Spectrogram Dataset Training/'
testing_path = r'C:\Users\brenn\Documents\School\Fourth Year\ELEC 490_Electrical Engineering Project\Dataset(Current)\Spectrogram Dataset Testing/'

testing_dataset = Tensor_images.dataset_to_tensors(testing_path, 300, 300)

model = tf.keras.models.load_model(r'C:\Users\brenn\Documents\GitHub\Music-Genre-Recognition-using-a-Machine-Learning-Approach\model_v1')

# Check its architecture
model.summary()

print('TESTING')

test_loss, test_acc = model.evaluate(testing_dataset, verbose=1)

img_file = r'C:\Users\brenn\Documents\School\Fourth Year\ELEC 490_Electrical Engineering Project\Dataset(Current)\FMA dataset\Spectrogram/'

fma_dataset = Tensor_images.dataset_to_tensors(img_file, 1, 1)

print(len(fma_dataset))
# Create compare array
compare_array = []
for i in range(26):
	compare_array.append(1)
for i in range(3):
	compare_array.append(2)
for i in range(750):
	compare_array.append(5)
for i in range(67):
	compare_array.append(6)
for i in range(70):
	compare_array.append(7)
for i in range(508):
	compare_array.append(8)
for i in range(86):
	compare_array.append(9)
for i in range(489):
	compare_array.append(10)




prediction = model.predict(fma_dataset)

predict_array = []
for row in prediction:
	predict_array.append(np.argmax(row) + 1)
	#print(np.argmax(row) + 1)

print(len(predict_array))
print(predict_array)
print(compare_array)


count = 0
for x in range (0,1999):
	if predict_array[x] == compare_array[x]:
		count = count + 1

print(count)
print(count/1999)