# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.python.framework import graph_util


def freeze_graph(input_meta_path, output_pb_path, output_node_names):
    if input_meta_path[-5:] != '.meta':
        input_meta_path += '.meta'

    saver = tf.train.import_meta_graph(input_meta_path, clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # print the following two loops to find `output_node_names`
    # for node in input_graph_def.node:
    #     print(node.name)

    # for operation in graph.get_operations():
    #     print(operation.name, operation.values())

    with tf.Session() as sess:
        saver.restore(sess, os.path.splitext(INPUT_META_PATH)[0])
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def, output_node_names.split(',')
        )

        os.makedirs(os.path.split(output_pb_path)[0], exist_ok=True)

        with tf.gfile.GFile(output_pb_path, "wb") as fid:
            fid.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        # print the following loop to check output nodes
        # for node in output_graph_def.node:
        #     print(node.name)


if __name__ == '__main__':
    INPUT_META_PATH = r'.\saved_model\model-938'
    OUTPUT_PB_PATH = r'.\saved_pb_model\model.pb'
    OUTPUT_NODE_NAMES = 'Softmax'
    freeze_graph(INPUT_META_PATH, OUTPUT_PB_PATH, OUTPUT_NODE_NAMES)
