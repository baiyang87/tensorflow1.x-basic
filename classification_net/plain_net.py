# -*- coding: utf-8 -*-
from basic_layers.tensorflow import conv_bn_relu
from basic_layers.tensorflow import dense
from basic_layers.tensorflow import batch_normalization
from basic_layers.tensorflow import maxpool
from basic_layers.tensorflow import global_average_pool
from basic_layers.tensorflow import softmax


class PlainNet:
    """
    Classification net with plain conv blocks as backbone
    """

    def __init__(self):
        self.config = {
            'block_filters': [32, 64, 128, 256, 512],
            'block_conv_nums': [2, 2, 2, 2, 2],
            'num_classes': 10,
            'use_bn': True,
        }
        self.use_bn = self.config.get('use_bn')
        self.block_filters = self.config.get('block_filters')
        self.block_conv_nums = self.config.get('block_conv_nums')
        self.training = True

    def build_blocks(self, inputs, out_filters, block_conv_num):
        """
        Build plain conv blocks

        Parameters
        ----------
        inputs: Input tensor
        out_filters: Number of output filters
        block_conv_num: Number of conv layers in block
        """
        if block_conv_num < 1:
            raise ValueError('block_conv_num must be >= 1')

        outputs = inputs
        for _ in range(block_conv_num):
            outputs = conv_bn_relu(outputs, out_filters, 3, 1, 1,
                                   use_bn=self.use_bn, training=self.training)
        return outputs

    def forward(self, inputs):
        """
        Forward process

        Parameters
        ----------
        inputs: Input tensor
        """
        outputs = inputs
        for i, filters in enumerate(self.block_filters):
            outputs = self.build_blocks(outputs, filters,
                                        self.block_conv_nums[i])

            if i != len(self.block_filters) - 1:
                outputs = maxpool(outputs, pool_size=2)
            else:
                outputs = global_average_pool(outputs)

        if self.use_bn:
            outputs = batch_normalization(outputs, training=self.training)

        outputs = dense(outputs, self.config.get('num_classes'))
        outputs = softmax(outputs)
        return outputs
