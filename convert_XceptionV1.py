import caffe
import cv2
import numpy as np

from keras.applications.xception import Xception
import keras

import keras2caffe

#TensorFlow backend uses all GPU memory by default, so we need limit
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

caffe.set_mode_gpu()

#converting

keras_model = Xception(input_shape=(299, 299, 3),include_top=True, weights='imagenet')


caffe_proto='XceptionV1.prototxt'
caffe_weights='XceptionV1.caffemodel'

keras2caffe.convert(keras_model, caffe_proto, caffe_weights)

#testing the model

net  = caffe.Net(caffe_proto, caffe_weights, caffe.TEST)

img = cv2.imread('bear.jpg')
img = cv2.resize(img, (299, 299))
img = img[...,::-1]  #RGB 2 BGR

data = np.array(img, dtype=np.float32)
data = data.transpose((2, 0, 1))
data.shape = (1,) + data.shape

data /= 128
data -= 1.0

net.blobs['data'].data[...] = data

out = net.forward()
pred = out['predictions']

prob = np.max(pred)
cls = pred.argmax()

lines=open('synset_words.txt').readlines()
print prob, cls, lines[cls]

