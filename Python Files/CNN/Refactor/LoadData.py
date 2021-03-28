import tensorflow as tf
import tensorflow.keras.preprocessing.image as image
import os
import numpy as np

def img_to_tensor(file, label):
    """
    Convert a single image into a tensor representation

    :param file: file to be converted into a tensor representation
    :param label: Class label for file
    :return:
    """
    # Get image from file
    img_str = tf.io.read_file(file)
    img_decode = tf.image.decode_jpeg(img_str, channels=3)
    # Return tensor for the image
    img = tf.cast(img_decode, tf.float32) / 255
    return img, label

# Parse function for tensor dataset creation
def set_shape(value, label):
    """
    Set shape of each tensor

    :param value:
    :param label: Class label for value
    :return:
    """
    # Set shape for image
    value.set_shape((370, 497, 3))
    return value, label


def get_dataset_files(data_loc):
    """
    Get the list of files to be used when converting to tensor

    :param data_loc: Location of the dataset directory
    :return: list of data filenames and labels
    """
    # Array to hold filenames of images
    dataset_filenames = []
    dataset_labels = []

    genre_count = 0
    # For each genre folder
    for genre_folder in os.listdir(data_loc):
        genre_loc = data_loc + genre_folder + '/'
        print(genre_loc)
        # For each image
        for file in os.listdir(genre_loc):
            dataset_filenames.append(genre_loc + file)
            dataset_labels.append(genre_count)
        genre_count = genre_count + 1

    return dataset_filenames, dataset_labels

def data_to_tensors(file_name):
    """

    :param file_name: Location of file
    :return:
    """
    #tensor, label = img_to_tensor(file_name, 1)

    img = image.load_img(file_name, target_size=(370, 497, 3))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    print(img)

    return img


def dataset_to_tensors(data_loc, batch_size, data_num):
    """
    Create a tensor dataset from the specified directory containing the spectrogram images

    :param data_loc: Directory containing the dataset split in folder
    :param batch_size: Batch size for the dataset for training. Specified here since tf dataset has attribute batch size
    :param data_num: Number of images in the dataset. Used to shuffle the data
    :return: Tensor dataset
    """

    dataset_filenames, dataset_labels = get_dataset_files(data_loc)

    # Create tensor from list of filenames and labels
    dataset_filenames = tf.convert_to_tensor(dataset_filenames, dtype=tf.string)
    dataset_labels = tf.convert_to_tensor(dataset_labels, dtype=tf.int32)

    # Merge files and labels in tensor
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

    return dataset
