import tensorflow as tf
import numpy as np
import LoadData
import Stereo_Mono_Conversion
import Spectogram_Conversion
import tkinter as tk
import ntpath
from PIL import ImageTk, Image



def main():
	def predict():
		result.configure(text="Analyzing...")

		inpt = txt.get()
		head, tail = ntpath.split(inpt)
		head = head + '/'

		wav_path = head
		wav_file_name = tail

		genre_array = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
		# Convert to mono
		mono_file = Stereo_Mono_Conversion.Stereo_Mono_Conversion(wav_path, wav_file_name)

		# Create spectrogram
		spectrogram = Spectogram_Conversion.spectrogram_conversion(wav_path + 'mono/', wav_path,
																   'mono_' + wav_file_name)

		tensor = LoadData.data_to_tensors(spectrogram)

		print(tensor)

		model = tf.keras.models.load_model('model_v1')

		# Check its architecture
		model.summary()

		prediction = model.predict_step(tensor)

		max_index = np.argmax(prediction)
		print(max_index)
		print(genre_array[max_index])

		outPut = "Music Genre Classification:\n" + genre_array[max_index]

		result.configure(text=outPut, font=("TkDefaultFont", 10, "bold"))

		path = spectrogram

		img = ImageTk.PhotoImage(Image.open(path).resize((250, 185)))

		panel.configure(image=img)
		panel.image=img

		window.geometry('650x500')


	window = tk.Tk()
	window.title("Genre Detector")

	window.geometry('650x400')

	ttl = tk.Label(window, text="Music Genre Detection Using a Machine Learning Approach", font=("TkDefaultFont", 14, "bold"))
	ttl.grid(column=1, row=0, columnspan=2)

	space = tk.Label(window, text=" ")
	space.grid(column=1, row=1, columnspan=2)

	desc = tk.Label(window, text="This program uses a Convolutional Neural Network\nto determine the musical genre of "
								 "a specified\naudio file. The file is first converted into a\nspectrogram image "
								 "before a prediction is made\nusing the CNN model. The model was trained on\nthe "
								 "GTZAN dataset which contains 1000 different\naudio samples.\n\nTo use this program, "
								 "simply enter the path of\nthe \'.wav\' audio file and click \"Detect\".")
	desc.grid(column=1,row=2, rowspan=3, sticky="w", padx=10)

	lbl = tk.Label(window, text="Enter File Path: ")
	lbl.grid(column=2, row=2, sticky="w", padx=10)

	txt = tk.Entry(window, width=50)
	txt.grid(column=2, row=3, sticky="n", padx=10)

	btn = tk.Button(window, text="Detect", command=predict)
	btn.grid(column=2, row=4, sticky="ne", padx=10)

	result = tk.Label(window, text="")
	result.grid(column=1, row=5, columnspan=2, pady=20)

	panel = tk.Label(window, image='')
	panel.grid(column=1, row=6, columnspan=2)

	window.mainloop()



if __name__ == '__main__':
	main()

