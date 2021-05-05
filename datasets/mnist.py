# -*- coding: utf-8 -*-
import os
import struct
import numpy as np

from datasets.data_utils import get_filenames_from_urls
from datasets.data_utils import download
from datasets.data_utils import extract_gz

URLS = {
    'train_data':
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'train_labels':
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'test_data':
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels':
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
}

DECOMPRESSED_FILENAMES = {
    'train_data': 'train-images.idx3-ubyte',
    'train_labels': 'train-labels.idx1-ubyte',
    'test_data': 't10k-images.idx3-ubyte',
    'test_labels': 't10k-labels.idx1-ubyte',
}


def load(folder=None):
    """
    Load mnist data from folder. If the data does NOT exist then download and
    decompress it.
    Data url: http://yann.lecun.com/exdb/mnist/

    Parameters
    ----------
    folder: Folder where to store mnist data and labels

    Returns
    -------
    data: Parsed mnist data and labels in dict type
        data = {
            'train_data': (60000, 28, 28) numpy array,
            'train_labels': (60000,) numpy array,
            'test_data': (10000, 28, 28) numpy array,
            'test_labels': (10000,) numpy array,
        }
    """
    if folder is None:
        folder = '.'
    os.makedirs(folder, exist_ok=True)

    data = {}
    filenames = get_filenames_from_urls(URLS)
    for name in filenames.keys():
        # dowanload and decompress data, if needed
        downloaded_filename = filenames.get(name)
        downloaded_filename = os.path.join(folder, downloaded_filename)
        decompressed_filename = DECOMPRESSED_FILENAMES.get(name)
        decompressed_filename = os.path.join(folder, decompressed_filename)

        if not os.path.exists(decompressed_filename):
            if not os.path.exists(downloaded_filename):
                url = URLS.get(name)
                download(url, folder)

            extract_gz(downloaded_filename, folder, decompressed_filename)

        # parse data
        if 'data' in name:
            data[name] = parse_mnist_images(decompressed_filename)
        elif 'labels' in name:
            data[name] = parse_mnist_labels(decompressed_filename)

    return data


def parse_mnist_images(filename):
    """
    Parse mnist image file and output the images as a uint8 numpy array, shape
    is (image_number, rows, cols)

    Parameters
    ----------
    filename: Filename of decompressed image file

    Returns
    -------
    images: The parsed images as a uint8 numpy array
    """
    with open(filename, 'rb') as fid:
        file_content = fid.read()
        item_number = struct.unpack('>i', file_content[4:8])[0]
        rows = struct.unpack('>i', file_content[8:12])[0]
        cols = struct.unpack('>i', file_content[12:16])[0]
        # 'item_number * rows * cols' is the number of bytes
        images = struct.unpack(
            '>%dB' % (item_number * rows * cols), file_content[16:])
        images = np.uint8(np.array(images))  # 1D
        images = np.reshape(images, [-1, rows, cols])
    return images


def parse_mnist_labels(filename):
    """
    Parse mnist label file and output the labels as a int32 numpy array, shape
    is (image_number,)

    Parameters
    ----------
    filename: Filename of decompressed label file

    Returns
    -------
    labels: The parsed labels as a int32 numpy array
    """
    with open(filename, 'rb') as fid:
        file_content = fid.read()
        item_number = struct.unpack('>i', file_content[4:8])[0]
        # 'item_number' is the number of bytes
        labels = struct.unpack('>%dB' % item_number, file_content[8:])
        labels = np.array(labels)
    return labels
