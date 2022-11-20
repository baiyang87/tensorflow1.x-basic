# -*- coding: utf-8 -*-
import os
import numpy as np
from urllib import request
import gzip
import tarfile


def download(url, folder=None, filename=None):
    """
    Download url to folder and store with filename

    Parameters
    ----------
    url: Web url of file to be downloaded
    folder: Folder where to store, default is '.'
    filename: Filename of the downloaded file, default is the last fragment of
        url
    """
    if folder is None:
        folder = '.'
    os.makedirs(folder, exist_ok=True)

    if filename is None:
        filename = os.path.split(url)[-1]

    full_name = os.path.join(folder, filename)
    if not os.path.exists(full_name):
        print("'%s' does NOT exist, downloading from %s" %
              (filename, url))
        request.urlretrieve(url, full_name)


def get_filenames_from_urls(urls):
    """
    Get filenames from urls by extracting the last fragment of each url

    Parameters
    ----------
    urls: Web urls, dict type

    Returns
    -------
    filenames: Filenames obtained from urls, dict type
    """
    filenames = {}
    for name in urls.keys():
        url = urls.get(name)
        filenames[name] = os.path.split(url)[-1]
    return filenames


def make_one_hot_labels(labels, num_classes, dtype=np.int32):
    """
    Transform classification labels represented by indices into one-hot labels,
    where the locations in `indices` take value 1 while all other locations
    take value 0

    Parameters
    ----------
    labels: labels represented by indices
    num_classes: number of classes
    dtype: output data type

    Returns
    -------
    one_hot_labels: one-hot encoded labels
    """
    one_hot_labels = (labels[:, None] == np.arange(num_classes))
    one_hot_labels = one_hot_labels.astype(dtype)
    return one_hot_labels


def extract_gz(gz_path, decompressed_file_path=None):
    """
    Extract gzip (.gz) file

    Parameters
    ----------
    gz_path: Path of .gz file
    decompressed_file_path: The filename of decompressed .gz file, default is
        the same as .gz file but without .gz extension
    """
    if decompressed_file_path is None:
        decompressed_file_path = gz_path.replace('.gz', '')

    folder = os.path.split(decompressed_file_path)[0]
    os.makedirs(folder, exist_ok=True)

    print("extract '%s' to get '%s'" % (gz_path, decompressed_file_path))
    with gzip.GzipFile(gz_path) as gz_file:
        open(decompressed_file_path, "wb+").write(gz_file.read())


def extract_tar(tar_path, folder=None):
    """
    Extract .tar file, including .tar.gz, .tar.bz2 et al.

    Parameters
    ----------
    tar_path: Path of .tar file
    folder: Folder where to decompress .tar file, default is the same folder as
        tar_path
    """
    if folder is None:
        folder = os.path.split(tar_path)[0]
    os.makedirs(folder, exist_ok=True)

    print("extract '%s' at %s" % (tar_path, folder))
    with tarfile.open(tar_path) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=folder)
