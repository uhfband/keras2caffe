import caffe
import cv2
import numpy as np

#TensorFlow backend uses all GPU memory by default, so we need limit
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

import keras2caffe

import sys
sys.path.append('/home/data/keras-models/keras-inceptionV4')

import inception_v4
#import evalute_image

#converting

keras_model = inception_v4.create_model(weights='imagenet', include_top=True, dropout_prob=0.8)

keras2caffe.convert(keras_model, 'InceptionV4.prototxt', 'InceptionV4.caffemodel')

#testing the model

caffe.set_mode_gpu()
net  = caffe.Net('InceptionV4.prototxt', 'InceptionV4.caffemodel', caffe.TEST)

img = cv2.imread('bear.jpg')
#img = evaluate_image.central_crop(im, 0.875)

img = cv2.resize(img, (299, 299))
img = img[...,::-1]  #RGB 2 BGR

data = np.array(img, dtype=np.float32)
data = data.transpose((2, 0, 1))
data.shape = (1,) + data.shape

#data -= 128
#data /= 128
data /= 256
data -= 1.0
#data = np.divide(data, 255.0)
#data = np.subtract(data, 1.0)
#data = np.multiply(data, 2.0)

net.blobs['data'].data[...] = data

out = net.forward()
preds = out['dense_1']

classes = eval(open('class_names.txt', 'r').read())
print("Class is: " + classes[np.argmax(preds)-1])
print("Certainty is: " + str(preds[0][np.argmax(preds)]))


