# -*- coding: utf-8 -*-
import tensorflow as tf
from datasets import load_mnist

DATA_FOLDER = r'.\data'


def load_data(folder):
    data = load_mnist(folder)
    train_data = data.get('train_data')
    train_labels = data.get('train_labels')
    test_data = data.get('test_data')
    test_labels = data.get('test_labels')
    return train_data, train_labels, test_data, test_labels


def train():
    # load data
    train_data, train_labels, test_data, test_labels = load_data(DATA_FOLDER)
    print(train_data.shape, train_labels.shape,
          test_data.shape, test_labels.shape)

    # train

    # test

    # save model
    pass


if __name__ == '__main__':
    train()
