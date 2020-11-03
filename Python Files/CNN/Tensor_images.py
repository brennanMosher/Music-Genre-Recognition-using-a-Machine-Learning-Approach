import tensorflow as tf
import os

'''
Contains function to convert dataset directory to tensor representation

dataset_to_tensors() should be called from other files

Do not use img_to_tensor() or set_shape() outside of this file

Parameters of dataset_to_tensors()
    data_loc: Location of data directory. Outside folder for all genre folders
    batch_size: Size of groupings for data. Experiment with changing this value
    data_num: Number of values in the dataset. Need this to know how many to include when shuffling. Can probably learn 
    this value internally. Try to do this in future
'''


# Parse function for Tensor mapping
def img_to_tensor(file, label):
    # file input is a list of filenames (including path)
    # Get image from file
    img_str = tf.io.read_file(file)
    # channels=3 returns RGB value (channels=2 returns grayscale)
    img_decode = tf.image.decode_jpeg(img_str, channels=3)
    # Return tensor for the image
    img = tf.cast(img_decode, tf.float32) / 255
    return img, label

# Parse function for tensor dataset creation
def set_shape(value, label):
    # Set shape for image
    value.set_shape((370, 497, 3))
    return value, label

# Create dataset from img files
def dataset_to_tensors(data_loc, batch_size, data_num):
    # Array to hold filenames of images
    dataset_filenames = []
    dataset_labels = []

    genre_count = 0
    # For each genre folder
    for genre_folder in os.listdir(data_loc):
        genre_loc = data_loc + genre_folder + '/'
        # For each image
        for file in os.listdir(genre_loc):
            dataset_filenames.append(genre_loc + file)
            dataset_labels.append(genre_count)
        genre_count = genre_count + 1

    # Create tensor from list of filenames and labels
    dataset_filenames = tf.convert_to_tensor(dataset_filenames, dtype=tf.string)
    dataset_labels = tf.convert_to_tensor(dataset_labels, dtype=tf.int32)

    # Merge tensors above
    dataset = tf.data.Dataset.from_tensor_slices((dataset_filenames, dataset_labels))
    print(dataset)

    # Shuffle the data so that same labels don't appear in batch
    dataset = dataset.shuffle(data_num)
    print(dataset)

    # Parse the data to return actual image values
    dataset = dataset.map(img_to_tensor)
    print(dataset)
    dataset = dataset.map(set_shape)
    print(dataset)

    # Set batch size within tensor instead of model.fit() function
    dataset = dataset.batch(batch_size)
    print(dataset)

    # Printing for visualization of tensor
    #np_list = list(dataset.as_numpy_iterator())
    #print(np_list[0])
    #print(len(np_list))

    return dataset