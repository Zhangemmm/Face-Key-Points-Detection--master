'''
__author__ =='syy'
关键点回归预测
'''
import sys
import numpy
import os
sys.path.append("F:/caffe-windows/python")
import caffe
pic_dir = "E:/dataset/FacialPoints/pyproject/save/"
image_list = "E:/dataset/FacialPoints/pyproject/test_1.txt"
WEIGHTS_FILE = 'E:/dataset/FacialPoints/model/snapshot_iter_800000.caffemodel'
DEPLOY_FILE = 'E:/dataset/FacialPoints/deploy.prototxt'
IMAGE_SIZE = (40, 40)
MEAN_VALUE = 128

caffe.set_mode_cpu()
net = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)
net.blobs['data'].reshape(1, 1, *IMAGE_SIZE)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', numpy.array([MEAN_VALUE]))
transformer.set_raw_scale('data', 255)
image_list = sys.argv[1]
with open(image_list, 'r') as f:
    lines = f.readlines()
for line in lines:
    filename = line[0]
    print(filename)
    image = caffe.io.load_image(os.path.join(pic_dir, filename), False)
    image = caffe.io.load_image(os.path.join(pic_dir, filename), False)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    print(output)
    # score = output['pred'][0][0]
    # print('The predicted score for {} is {}'.format("filename", score))