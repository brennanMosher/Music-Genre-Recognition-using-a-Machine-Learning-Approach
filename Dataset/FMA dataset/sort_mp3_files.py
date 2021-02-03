import os
from eyed3 import id3
import eyed3
import shutil

unsorted_loc = r"C:\Users\brenn\fma\data\fma_small/"
sorted_loc = r"C:\Users\brenn\fma\data\Sorted/"

eyed3.log.setLevel("ERROR")

for folder in os.listdir(unsorted_loc):
	for file in os.listdir(unsorted_loc + folder):
		file_loc = unsorted_loc + folder + "/" + file
		tag = id3.Tag()
		tag.parse(file_loc)
		if "blues" in str(tag.genre).lower():
			print("copying...")
			dst = sorted_loc + 'blues/' + file
			shutil.copy2(file_loc, dst)
			print('Done')
		if "classical" in str(tag.genre).lower():
			print("copying...")
			dst = sorted_loc + 'classical/' + file
			shutil.copy2(file_loc, dst)
			print('Done')
		if "country" in str(tag.genre).lower():
			print("copying...")
			dst = sorted_loc + 'country/' + file
			shutil.copy2(file_loc, dst)
			print('Done')
		if "disco" in str(tag.genre).lower():
			print("copying...")
			dst = sorted_loc + 'disco/' + file
			shutil.copy2(file_loc, dst)
			print('Done')
		if "hip-hop" in str(tag.genre).lower():
			print("copying...")
			dst = sorted_loc + 'hiphop/' + file
			shutil.copy2(file_loc, dst)
			print('Done')
		if "jazz" in str(tag.genre).lower():
			print("copying...")
			dst = sorted_loc + 'jazz/' + file
			shutil.copy2(file_loc, dst)
			print('Done')
		if "metal" in str(tag.genre).lower():
			print("copying...")
			dst = sorted_loc + 'metal/' + file
			shutil.copy2(file_loc, dst)
			print('Done')
		if "pop" in str(tag.genre).lower():
			print("copying...")
			dst = sorted_loc + 'pop/' + file
			shutil.copy2(file_loc, dst)
			print('Done')
		if "reggae" in str(tag.genre).lower():
			print("copying...")
			dst = sorted_loc + 'reggae/' + file
			shutil.copy2(file_loc, dst)
			print('Done')
		if "rock" in str(tag.genre).lower():
			print("copying...")
			dst = sorted_loc + 'rock/' + file
			shutil.copy2(file_loc, dst)
			print('Done')
