# -*- coding: utf-8 -*-
from basic_layers.tensorflow import conv
from basic_layers.tensorflow import conv_bn_relu
from basic_layers.tensorflow import maxpool
from basic_layers.tensorflow import upsample
from basic_layers.tensorflow import concat
from basic_layers.tensorflow import softmax


class PlainUnet(object):
    """
    Unet with plain conv blocks as backbone
    """

    def __init__(self):
        self.config = {
            'block_filters': [32, 64, 128, 256, 512],
            'block_conv_nums': [2, 2, 2, 2, 2],
            'output_channel': 10,
            'use_bn': True,
            'interpolation_type': 'bilinear',
        }
        self.use_bn = self.config.get('use_bn')
        self.interpolation_type = self.config.get('interpolation_type')
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
        # encoder process
        outputs = inputs
        encoders = []
        for i, filters in enumerate(self.block_filters):
            outputs = self.build_blocks(outputs, filters,
                                        self.block_conv_nums[i])

            if i != len(self.block_filters) - 1:
                encoders.append(outputs)
                outputs = maxpool(outputs, pool_size=2)

        # decoder process
        encoder_num = len(encoders)
        for k, encoder in enumerate(encoders[::-1]):
            i = encoder_num - k - 1
            filters = int(encoder.get_shape()[-1])
            outputs = conv_bn_relu(outputs, filters, 3, 1, 1,
                                   use_bn=self.use_bn, training=self.training)
            outputs = upsample(outputs, 2, self.interpolation_type)
            outputs = concat([encoder, outputs])
            outputs = self.build_blocks(outputs, filters,
                                        self.block_conv_nums[i])

        outputs = conv(outputs, self.config.get('output_channel'))
        outputs = softmax(outputs)
        return outputs
