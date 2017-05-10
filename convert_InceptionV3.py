import caffe
import cv2
import numpy as np

from keras.applications.inception_v3 import InceptionV3

import keras2caffe


#converting

keras_model = InceptionV3(weights='imagenet', include_top=True)

keras2caffe.convert(keras_model, 'deploy.prototxt', 'InceptionV3.caffemodel')


#testing the model

caffe.set_mode_gpu()
net  = caffe.Net('deploy.prototxt', 'InceptionV3.caffemodel', caffe.TEST)

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

