import pathlib

# Gets users paths to create training/testing paths
def get_data_path(training_loc, testing_loc):
	# Get path users path to file
	path = pathlib.Path().absolute()
	# Get source root path
	path = path.parents[2]
	# Cast path as string
	path = str(path)
	# Get training/testing paths to return
	training_path = path + training_loc
	testing_path = path + testing_loc

	return training_path, testing_path
