# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from datasets import load_mnist
from datasets import make_one_hot_labels
from classification_net.plain_net import PlainNet
from losses import softmax_cross_entropy

DATA_FOLDER = r'.\data'
NUM_CLASSES = 10
EPOCH = 10
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 32


def load_data(folder):
    data = load_mnist(folder)
    train_data = data.get('train_data')
    train_labels = data.get('train_labels')
    test_data = data.get('test_data')
    test_labels = data.get('test_labels')

    # expand_dims for data
    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    # make one-hot labels
    train_labels = make_one_hot_labels(train_labels, NUM_CLASSES)
    test_labels = make_one_hot_labels(test_labels, NUM_CLASSES)

    return train_data, train_labels, test_data, test_labels


def make_model():
    model = PlainNet()
    model.config['num_classes'] = NUM_CLASSES
    model.config['block_filters'] = [32, 64, 128]
    model.config['block_conv_nums'] = [2, 2, 2]
    model.config['use_bn'] = False
    return model


def train():
    # load data
    train_data, train_labels, test_data, test_labels = load_data(DATA_FOLDER)
    print(train_data.shape, train_labels.shape,
          test_data.shape, test_labels.shape)

    # model
    model = make_model()

    # placeholder
    data = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    labels = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

    # network settings
    outputs = model.forward(data)
    cross_entropy_loss = softmax_cross_entropy(labels, outputs)
    optimizer = tf.train.AdamOptimizer(0.001)
    train_step = optimizer.minimize(cross_entropy_loss)
    print(train_step)

    # train

    # test

    # save model
    pass


if __name__ == '__main__':
    train()
