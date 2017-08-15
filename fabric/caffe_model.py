#!/usr/bin/env python 

import imgutil
import os, sys
import pickle
import caffe
from config.PROJECT import config


num_chars = config['num_chars']
width = config['width']
height = config['height']
model_path = config['remote']['model']
proto_path =  config['remote']['proto']
revert_dict = config['remote']['rdict']

caffe.set_mode_cpu()
net_alnum = caffe.Net(proto_path, model_path, caffe.TEST)

transformer_alnum = caffe.io.Transformer({'data': net_alnum.blobs['data'].data.shape})
transformer_alnum.set_transpose('data', (2, 0, 1))
transformer_alnum.set_raw_scale('data', 1)

alnum_dict = {}
f = open( revert_dict, 'rb')
alnum_dict = pickle.load(f)
f.close()


def predict(post_input):
    query_file = post_input.get('image').value
    query_img = imgutil.binary_to_image(query_file)
    query_img = imgutil.normalize(query_img)
    #remove_alpha
    #query_img = imgutil.remove_alpha(query_img)
    net_alnum.blobs['data'].data[...] = transformer_alnum.preprocess('data', query_img)
    net_alnum.blobs['data'].reshape(1, 3, height, width)
    out = net_alnum.forward()
    output_str = ''
    for idx in range(num_chars):
        output_str += alnum_dict[out['prob{0}'.format(idx)].argmax()]
    return output_str

