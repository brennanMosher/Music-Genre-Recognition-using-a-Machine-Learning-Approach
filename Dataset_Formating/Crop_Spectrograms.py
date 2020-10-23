import os
from PIL import Image

file_loc = r'C:\Users\brenn\Documents\GitHub\Music-Genre-Recognition-using-a-Machine-Learning-Appraoch\Dataset' \
           r'\genres_original/'
folder_name = '/Spectograms/'

# Each genre folder
for folder in os.listdir(file_loc):
    print(folder)
    # Train and Test folder
    for type in os.listdir(file_loc + folder):
        print(type)
        spectrogram_loc = file_loc + folder + '/' + type+ folder_name
        # Each Spectrogram
        for image in os.listdir(spectrogram_loc):
            img = Image.open(spectrogram_loc + image)

            print(image)
            #img.show()

            # (left, upper, rigth, lower)
            crop_box = (80,58,577,428)
            img1 = img.crop(crop_box)
            img1.save(spectrogram_loc + image)
