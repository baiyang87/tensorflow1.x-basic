# -*- coding: utf-8 -*-
from basic_layers.tensorflow import conv_bn_relu
from basic_layers.tensorflow import batch_normalization
from basic_layers.tensorflow import conv
from basic_layers.tensorflow import dense
from basic_layers.tensorflow import concat
from basic_layers.tensorflow import relu
from basic_layers.tensorflow import maxpool
from basic_layers.tensorflow import global_average_pool
from basic_layers.tensorflow import dropout


class DenseNet:
    """
    Classification net with with dense blocks as backbone
    """

    def __init__(self):
        self.config = {
            'block_filters': [32, 64, 128, 256, 512],
            'block_dense_nums': [2, 4, 8, 8, 8],
            'num_classes': 10,
            'use_bn': True,
            'growth_rates': [0, 16, 16, 32, 32],
            'bottleneck': True,
            'bottleneck_factor': 4,
            'dropout_rate': 0.2,
        }
        self.training = True

    def conv_block(self, inputs, out_filters, ksize):
        """
        Pre-activated conv block (BN-ReLU-Conv)

        Parameters
        ----------
        inputs: Input tensor
        out_filters: Number of output filters
        ksize: Kernel size. One integer of tuple of two integers
        """
        use_bn = self.config.get('use_bn')
        use_bias = not use_bn

        outputs = inputs
        if use_bn:
            outputs = batch_normalization(outputs, training=self.training)
        outputs = relu(outputs)
        outputs = conv(outputs, out_filters, ksize=ksize, use_bias=use_bias)
        return outputs

    def build_one_dense_block(self, inputs, growth_rate):
        """
        Build one dense conv block

        Parameters
        ----------
        inputs: Input tensor
        growth_rate: number of channel growth for each dense block
        """
        if self.config.get('bottleneck'):
            shrinked_filters = self.config.get('bottleneck_factor') * \
                               growth_rate
            branch = self.conv_block(inputs, shrinked_filters, ksize=1)
            branch = self.conv_block(branch, growth_rate, ksize=3)
        else:
            branch = self.conv_block(inputs, growth_rate, ksize=3)

        if self.config.get('dropout_rate'):
            branch = dropout(branch, self.config.get('dropout_rate'))

        outputs = concat([inputs, branch])
        return outputs

    def build_dense_blocks(self, inputs, out_filters, block_dense_num,
                           growth_rate, first_blocks):
        """
        Build multiple residual blocks

        Parameters
        ----------
        inputs: Input tensor
        out_filters: Number of output filters
        block_dense_num: Number of dense blocks
        growth_rate: number of channel growth for each dense block
        first_blocks: Whether is the first_blocks (before first pooling layer).
            The first_blocks is not implemented by dense blocks but plain
            conv layers
        """
        use_bn = self.config.get('use_bn')

        if block_dense_num < 1:
            raise ValueError('block_dense_num must be >= 1')

        outputs = inputs
        if first_blocks:
            for _ in range(block_dense_num):
                outputs = conv_bn_relu(outputs, out_filters, 3, 1, 1,
                                       use_bn=use_bn, training=self.training)
        else:
            for _ in range(block_dense_num):
                outputs = self.build_one_dense_block(outputs, growth_rate)

        outputs = self.conv_block(outputs, out_filters, ksize=1)
        return outputs

    def forward(self, inputs):
        """
        Forward process

        Parameters
        ----------
        inputs: Input tensor

        Returns
        -------
        outputs: Predictions of network, shape is (batch, num_classes).
            The outputs are called logits for classification net because the
            softmax activations are not applied in this forward process
        """
        block_filters = self.config.get('block_filters')
        block_dense_nums = self.config.get('block_dense_nums')

        outputs = inputs
        for i, filters in enumerate(block_filters):
            first_blocks = i == 0
            growth_rate = self.config.get('growth_rates')[i]
            outputs = self.build_dense_blocks(outputs,
                                              filters,
                                              block_dense_nums[i],
                                              growth_rate,
                                              first_blocks)

            if i != len(block_filters) - 1:
                outputs = maxpool(outputs, pool_size=2)
            else:
                outputs = global_average_pool(outputs)

        outputs = dense(outputs, self.config.get('num_classes'))
        return outputs
