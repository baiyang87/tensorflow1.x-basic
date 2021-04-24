# -*- coding: utf-8 -*-
from basic_layers.tensorflow import conv_bn_relu
from basic_layers.tensorflow import dense
from basic_layers.tensorflow import maxpool
from basic_layers.tensorflow import add
from basic_layers.tensorflow import relu
from basic_layers.tensorflow import global_average_pool
from basic_layers.tensorflow import softmax


class ResNet(object):
    """
    Classification net with with residual blocks as backbone
    """

    def __init__(self):
        self.config = {
            'block_filters': [32, 64, 128, 256, 512],
            'block_res_nums': [2, 2, 2, 2, 2],
            'output_channel': 10,
            'use_bn': True,
            'interpolation_type': 'bilinear',
            'shrink_factor': 4,
            'bottleneck': False,
        }
        self.use_bn = self.config.get('use_bn')
        self.interpolation_type = self.config.get('interpolation_type')
        self.block_filters = self.config.get('block_filters')
        self.block_res_nums = self.config.get('block_res_nums')
        self.training = True

    def build_one_res_block(self, inputs, out_filters, shrink_factor):
        """
        Build one residual conv block

        Parameters
        ----------
        inputs: Input tensor
        out_filters: Number of output filters
        shrink_factor: shrink factor with respect to out_filters for
            intermediate conv layers
        """
        in_filters = int(inputs.get_shape()[-1])
        shrinked_filters = int(out_filters / shrink_factor)

        # identity branch: branch_a
        if in_filters == out_filters:
            branch_a = inputs
        else:
            branch_a = conv_bn_relu(inputs, out_filters, 1, 1, 1,
                                    use_bn=self.use_bn, use_relu=False,
                                    training=self.training)

        # conv branch: branch_b
        if self.config.get('bottleneck'):
            branch_b = conv_bn_relu(inputs, shrinked_filters, 1, 1, 1,
                                    use_bn=self.use_bn, training=self.training)
            branch_b = conv_bn_relu(branch_b, shrinked_filters, 3, 1, 1,
                                    use_bn=self.use_bn, training=self.training)
            branch_b = conv_bn_relu(branch_b, out_filters, 1, 1, 1,
                                    use_bn=self.use_bn, use_relu=False,
                                    training=self.training)
        else:
            branch_b = conv_bn_relu(inputs, shrinked_filters, 3, 1, 1,
                                    use_bn=self.use_bn, training=self.training)
            branch_b = conv_bn_relu(branch_b, out_filters, 3, 1, 1,
                                    use_bn=self.use_bn, use_relu=False,
                                    training=self.training)
        branch = add([branch_a, branch_b])
        branch = relu(branch)
        return branch

    def build_res_blocks(self, inputs, out_filters, block_res_num,
                         first_blocks):
        """
        Build multiple residual blocks

        Parameters
        ----------
        inputs: Input tensor
        out_filters: Number of output filters
        block_res_num: Number of residual blocks
        first_blocks: Whether the first_blocks. The first_blocks is not
            implemented by residual blocks but plain blocks
        """
        if block_res_num < 1:
            raise ValueError('block_conv_num must be >= 1')

        outputs = inputs
        if first_blocks:
            for _ in range(block_res_num):
                outputs = conv_bn_relu(outputs, out_filters, 3, 1, 1,
                                       use_bn=self.use_bn,
                                       training=self.training)
        else:
            shrink_factor = self.config.get('shrink_factor')
            for _ in range(block_res_num):
                outputs = self.build_one_res_block(outputs, out_filters,
                                                   shrink_factor)
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
            first_blocks = i == 0
            outputs = self.build_res_blocks(outputs, filters,
                                            self.block_res_nums[i],
                                            first_blocks)

            if i != len(self.block_filters) - 1:
                outputs = maxpool(outputs, pool_size=2)
            else:
                outputs = global_average_pool(outputs)

        outputs = dense(outputs, self.config.get('num_classes'))
        outputs = softmax(outputs)
        return outputs
