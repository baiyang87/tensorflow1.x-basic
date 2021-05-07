# -*- coding: utf-8 -*-
import tensorflow as tf
from classification_net.plain_net import PlainNet

if __name__ == '__main__':
    inputs = tf.ones(shape=(7, 256, 256, 3))
    net = PlainNet()
    net.config['num_classes'] = 20
    net.config['block_filters'] = [32, 64, 128]
    net.config['block_conv_nums'] = [2, 2, 2]
    outputs = net.forward(inputs)
    print(outputs)
