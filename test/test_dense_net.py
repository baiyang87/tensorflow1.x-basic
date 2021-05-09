# -*- coding: utf-8 -*-
import tensorflow as tf
from classification_net.dense_net import DenseNet

if __name__ == '__main__':
    inputs = tf.ones(shape=(7, 256, 256, 3))
    net = DenseNet()
    net.config['num_classes'] = 20
    net.config['block_filters'] = [32, 64, 128]
    net.config['block_dense_nums'] = [3, 6, 6]
    net.config['use_bn'] = False
    outputs = net.forward(inputs)
    print(outputs)

    print()
    for var in tf.trainable_variables():
        print(var)
