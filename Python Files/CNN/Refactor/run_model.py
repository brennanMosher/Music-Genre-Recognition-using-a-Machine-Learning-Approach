import LoadData
import MusicGenre


def init_params():
	"""
	Init parameters for the tensordatasets

	:return:
	"""
	batch_size = 10
	dataset_size = 700
	return batch_size, dataset_size

def main():
	# TODO simplify paths
	training_path = r"C:\Users\brenn\Documents\School\Fourth Year\ELEC 490_Electrical Engineering Project" \
					r"\Dataset(Current)\Spectrogram Dataset Training/"
	testing_path = r"C:\Users\brenn\Documents\School\Fourth Year\ELEC 490_Electrical Engineering Project" \
				   r"\Dataset(Current)\Spectrogram Dataset Testing/"

	batch_size, dataset_size= init_params()

	# Return dataset objects from tensor creation function
	training_dataset = LoadData.dataset_to_tensors(training_path, batch_size, dataset_size)
	testing_dataset = LoadData.dataset_to_tensors(testing_path, 300, 300)

	# TODO: Remove print statements
	print('dataset_size')
	print(len(training_dataset))

	# Validation split = 0.3
	split = int(len(training_dataset) * 0.3)
	# Skip and take are used to split the training dataset into a training and validation set
	validation_dataset = training_dataset.take(split)
	training_dataset = training_dataset.skip(split)

	print(len(training_dataset))
	print(len(validation_dataset))

	model = MusicGenre.MusicGenre()
	model.train(training_dataset, validation_dataset, epochs=50)
	test_loss, test_acc = model.test(testing_dataset)

	print(test_loss)
	print(test_acc)


if __name__ == '__main__':
	main()
