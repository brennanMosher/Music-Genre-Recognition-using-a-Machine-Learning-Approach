import tensorflow as tf
import numpy as np
import LoadData
import Stereo_Mono_Conversion
import Spectogram_Conversion


def main():
	wav_path = 'gui/'
	wav_file_name = 'blues.00070.wav'

	genre_array = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
	# Convert to mono
	mono_file = Stereo_Mono_Conversion.Stereo_Mono_Conversion(wav_path, wav_file_name)

	# Create spectrogram
	spectrogram = Spectogram_Conversion.spectrogram_conversion(wav_path+'mono/', wav_path, 'mono_'+wav_file_name)

	tensor = LoadData.data_to_tensors(spectrogram)

	print(tensor)

	model = tf.keras.models.load_model('model_v1')

	# Check its architecture
	model.summary()

	prediction = model.predict_step(tensor)

	max_index = np.argmax(prediction)
	print(max_index)
	print(genre_array[max_index])

if __name__ == '__main__':
	main()

