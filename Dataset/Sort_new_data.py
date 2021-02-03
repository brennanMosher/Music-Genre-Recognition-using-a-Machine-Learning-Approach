import os
from eyed3 import id3

unsorted_loc = r"C:\Users\brenn\fma\data\fma_small/"

txt_loc = unsorted_loc + 'genre.txt'
f = open(txt_loc, 'w', encoding="utf-8")


for folder in os.listdir(unsorted_loc):
	for file in os.listdir(unsorted_loc + folder):
		#print(file)
		file_loc = unsorted_loc + folder + '/' + file
		tag = id3.Tag()
		tag.parse(file_loc)
		print(tag.genre)
		f.write(str(tag.genre) + '\n')
f.close()
