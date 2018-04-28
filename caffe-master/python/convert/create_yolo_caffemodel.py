# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:10:21 2016

@author: xingw, Banus
"""
from __future__ import print_function, division

import argparse
import os

import numpy as np

if 'GLOG_minloglevel' not in os.environ:
    os.environ['GLOG_minloglevel'] = '2'  # suppress verbose Caffe output

import caffe


def transpose_matrix(array, rows, cols):
    """ transpose flattened matrix """
    return array.reshape((rows, cols)).transpose().flatten()


def load_parameter(weights, layer_data, transpose=False):
    """  load Caffe parameters from YOLO weights """
    shape = layer_data.shape
    size = np.prod(shape)
    if size > weights.size:
        raise ValueError(" Layer too big: required {} weights, available {}".
                         format(size, weights.size))

    if transpose:
        layer_data[...] = np.reshape(
            transpose_matrix(weights[:size], np.prod(shape[1:]), shape[0]), shape)
    else:
        layer_data[...] = np.reshape(weights[:size], shape)

    return size


def convert_weights(model_filename, yoloweight_filename, caffemodel_filename):
    """ convert YOLO weights to .caffemodel format given the caffe model """
    net = caffe.Net(model_filename, caffe.TEST)

    # read header to get the transpose flag
    weights_int = np.fromfile(yoloweight_filename, dtype=np.int32, count=4)
    # transpose flag, the first 4 entries are major, minor, revision and net.seen
    transp_flag = (weights_int[0] > 1000 or weights_int[1] > 1000)
    print('Transpose fc layers: {}'.format(transp_flag))

    # read the weights from YOLO file, skipping the header
    weights = np.fromfile(yoloweight_filename, dtype=np.float32)[4:]

    count = 0
    for name, layer in zip(net.top_names, net.layers):
        if name not in net.params.keys():  # layer without parameters
            continue
        if layer.type in ['BatchNorm', 'Scale']:
            continue   # handled within the convolutional layer

        print("  converting {0}".format(name))

        if   layer.type == 'Convolution':
            bn_name = "{0}_bn".format(name)
            scale_name = "{0}_scale".format(name)
            if bn_name in net.top_names:  # there is a batchnorm layer
                # YOLO stores bias, scale, rolling mean, rolling variance in
                # this order

                # set bias and scale on the ScaleLayer in Caffe
                count += load_parameter(weights[count:], net.params[scale_name][1].data)
                count += load_parameter(weights[count:], net.params[scale_name][0].data)
                # load mean and variance for BatchNormLayer in Caffe
                count += load_parameter(weights[count:], net.params[bn_name][0].data)
                count += load_parameter(weights[count:], net.params[bn_name][1].data)
                # use Scale Layer for channel-wise scel factor
                net.params[bn_name][2].data[...] = 1.0
            else:
                count += load_parameter(weights[count:], net.params[name][1].data) # conv bias
            # weights
            count += load_parameter(weights[count:], net.params[name][0].data)
        elif layer.type == 'InnerProduct':   # fc layer
            count += load_parameter(weights[count:], net.params[name][1].data) # bias
            count += load_parameter(weights[count:], net.params[name][0].data, transp_flag)
        elif layer.type == 'LocalConvolution':
            count += load_parameter(weights[count:], net.params[name][1].data) # bias
            count += load_parameter(weights[count:], net.params[name][0].data)
        else:
            print("WARNING: unknown type {} for layer {}".format(layer.type, name))

    if count != weights.shape[0]:  # some weights were left out
        raise ValueError(" Wrong number of weights: read {0}, used {1} (missing {2})".
                         format(weights.size, count, weights.size-count))
    print('Converted {0} weights.'.format(count))
    net.save(caffemodel_filename)


def main():
    """ script entry point """
    parser = argparse.ArgumentParser(description='Convert YOLO weights to Caffe.')
    parser.add_argument('model', type=str, help='Caffe model file')
    parser.add_argument('yolo_weights', type=str, help='YOLO weight file')
    parser.add_argument('output', type=str, help='converted .caffemodel')
    args = parser.parse_args()

    print('model file is {}'.format(args.model))
    print('weight file is {}'.format(args.yolo_weights))
    print('output caffemodel file is {}'.format(args.output))

    convert_weights(args.model, args.yolo_weights, args.output)


if __name__ == '__main__':
    main()
