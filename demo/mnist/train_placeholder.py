# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf

from datasets import load_mnist
from datasets import make_one_hot_labels
from classification_net.plain_net import PlainNet
from losses import softmax_cross_entropy
from metrics import logits_accuracy

# parameters
DATA_FOLDER = r'.\data'
SAVE_MODEL_PREFIX = r'.\saved_model\model'
NUM_CLASSES = 10
EPOCH = 2
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


def random_shuffle(data, labels):
    number = data.shape[0]
    random_index = np.arange(number)
    np.random.shuffle(random_index)
    data = data[random_index]
    labels = labels[random_index]
    return data, labels


def load_batch(data, labels, index_beg, batch_size):
    total_num = data.shape[0]
    index_end = min(index_beg + batch_size, total_num)
    batch_data = data[index_beg:index_end]
    batch_labels = labels[index_beg:index_end]
    return batch_data, batch_labels


def train():
    # load data
    train_data, train_labels, test_data, test_labels = load_data(DATA_FOLDER)
    num_train = train_data.shape[0]
    num_test = test_data.shape[0]

    # model
    model = make_model()

    # placeholder
    data = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    labels = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

    # network settings
    outputs = model.forward(data)
    cross_entropy_loss = softmax_cross_entropy(labels, outputs)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_loss)
    _, batch_correct_num, batch_total_num = logits_accuracy(labels, outputs)

    # initialization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saved_variables = tf.trainable_variables()
    saver = tf.train.Saver(var_list=saved_variables, max_to_keep=None)
    global_step = 0

    for i in range(EPOCH):
        print('-' * 20, 'epoch %d' % (i + 1), '-' * 20)

        # train
        train_data, train_labels = random_shuffle(train_data, train_labels)
        for index_beg in range(0, num_train, TRAIN_BATCH_SIZE):
            global_step += 1
            index_end = min(index_beg + TRAIN_BATCH_SIZE, num_train)
            batch_data, batch_labels = load_batch(train_data, train_labels,
                                                  index_beg, TRAIN_BATCH_SIZE)
            loss, _ = sess.run([cross_entropy_loss, train_step],
                               feed_dict={data: batch_data,
                                          labels: batch_labels})
            print('\rprogress = %0.2f%%, loss = %0.6f' % (
                index_end / num_train * 100, loss), end='')
        print()

        # test
        correct_num = 0
        total_num = 0
        for index_beg in range(0, num_test, TEST_BATCH_SIZE):
            batch_data, batch_labels = load_batch(test_data, test_labels,
                                                  index_beg, TEST_BATCH_SIZE)
            c_num, b_num = sess.run([batch_correct_num, batch_total_num],
                                    feed_dict={data: batch_data,
                                               labels: batch_labels})
            correct_num += c_num
            total_num += b_num
            print('\raccuracy = %0.4f' % (correct_num / total_num), end='')
        print()

        # save model
        os.makedirs(os.path.split(SAVE_MODEL_PREFIX)[0], exist_ok=True)
        saver.save(sess, SAVE_MODEL_PREFIX, global_step=global_step)


if __name__ == '__main__':
    train()
