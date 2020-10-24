import os
from PIL import Image

'''
Test file for cropping images
'''


file_loc = r'C:\Users\brenn\Desktop\Spectrogram/blues.00000.jpg'

im = Image.open(file_loc)

crop_box = (80, 58, 576, 427)
img1 = im.crop(crop_box)
img1.show()