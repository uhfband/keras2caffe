import caffe
import cv2
import numpy as np

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

import keras2caffe

#TensorFlow backend uses all GPU memory by default, so we need limit
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

#converting

keras_model = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=True)

keras2caffe.convert(keras_model, 'VGG16.prototxt', 'VGG16.caffemodel')

#testing the model

#caffe.set_mode_gpu()
net  = caffe.Net('VGG16.prototxt', 'VGG16.caffemodel', caffe.TEST)

img = cv2.imread('bear.jpg')
img = cv2.resize(img, (224, 224))
img = img[...,::-1]  #RGB 2 BGR

data = np.array(img, dtype=np.float32)
data = data.transpose((2, 0, 1))
data.shape = (1,) + data.shape

data -= 128

out = net.forward_all(data = data)
pred = out['predictions']
prob = np.max(pred)
cls = pred.argmax()
lines=open('synset_words.txt').readlines()
print prob, cls, lines[cls]
