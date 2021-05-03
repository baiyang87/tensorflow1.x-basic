# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np

from datasets.data_utils import get_filenames_from_urls
from datasets.data_utils import download
from datasets.data_utils import extract_tar

URLS = {
    'data': r'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
}

DECOMPRESSED_FOLDER = r'cifar-10-batches-py'

DECOMPRESSED_FILENAMES = {
    'train': ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
              'data_batch_5'],
    'test': ['test_batch'],
}


def load(folder=None):
    """
    Load CIFAR10 data from folder. If it does NOT exist then download and
    decompress it.

    Parameters
    ----------
    folder: Folder where to store CIFAR10 data and labels

    Returns
    -------
    data: Parsed mnist data and labels in dict type
        data = {
            'train_data': (50000, 32, 32, 3) numpy array,
            'train_labels': (50000,) numpy array,
            'test_data': (10000, 32, 32, 3) numpy array,
            'test_labels': (10000,) numpy array,
        }
    """
    if folder is None:
        folder = '.'
    original_folder = folder
    parent_folder = os.path.split(original_folder)[0]
    sub_folder = os.path.join(original_folder, DECOMPRESSED_FOLDER)

    # check existence and completeness of data, download or/and decompress the
    # data if needed
    filenames = get_filenames_from_urls(URLS)
    downloaded_filename = filenames.get('data')

    if exist_data(original_folder):
        folder = original_folder
    elif exist_data(sub_folder):
        folder = sub_folder
    elif extract_data(os.path.join(original_folder, downloaded_filename)):
        folder = sub_folder
    elif extract_data(os.path.join(parent_folder, downloaded_filename)):
        folder = original_folder
    else:
        download(URLS.get('data'), original_folder, downloaded_filename)
        extract_tar(os.path.join(original_folder, downloaded_filename))
        folder = sub_folder

    # parse data
    data = {
        'train_data': [],
        'train_labels': [],
        'test_data': [],
        'test_labels': [],
    }
    for filename in DECOMPRESSED_FILENAMES.get('train'):
        path = os.path.join(folder, filename)
        temp_data, temp_labels = parse_one_data_file(path)
        data['train_data'].append(temp_data)
        data['train_labels'].append((temp_labels))

    for filename in DECOMPRESSED_FILENAMES.get('test'):
        path = os.path.join(folder, filename)
        temp_data, temp_labels = parse_one_data_file(path)
        data['test_data'].append(temp_data)
        data['test_labels'].append(temp_labels)

    # concatenate data
    data['train_data'] = np.concatenate(data.get('train_data'), axis=0)
    data['train_labels'] = np.concatenate(data.get('train_labels'), axis=0)
    data['test_data'] = np.concatenate(data.get('test_data'), axis=0)
    data['test_labels'] = np.concatenate(data.get('test_labels'), axis=0)
    return data


def unpickle(filename):
    """
    Parse CIFAR10 data.
    Return a dict containing {data, filenames, labels, batch_label}
    """
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def exist_data(folder):
    """
    Check existence and completeness of decompressed data in folder

    Parameters
    ----------
    folder: Folder where to store the decompressed CIFAR10 data
    """
    for name in DECOMPRESSED_FILENAMES.keys():
        filename_list = DECOMPRESSED_FILENAMES.get(name)
        for filename in filename_list:
            full_name = os.path.join(folder, filename)
            if not os.path.exists(full_name):
                return False
    return True


def extract_data(path):
    """
    Extract data if path exists. Return True if successfully extracted else
    False

    Parameters
    ----------
    path: Path of downloaded .tar.gz file
    """
    if os.path.exists(path):
        extract_tar(path)
        return True
    return False


def parse_one_data_file(path):
    """
    Parse one data file to obtain data and labels

    Parameters
    ----------
    path: Path of data file, such as 'xxxx/data_batch_1'
    """
    data_dict = unpickle(path)
    # type of key in data_dict is not 'str' but 'bytes', a prefix of 'b'
    # should be ahead of key in order to get value
    data = data_dict.get(b'data')
    labels = data_dict.get(b'labels')
    # default cifar-10 data encoding is channel first
    data = np.reshape(data, [-1, 3, 32, 32])
    # transpose to channel last
    data = np.transpose(data, [0, 2, 3, 1])

    return data, labels
