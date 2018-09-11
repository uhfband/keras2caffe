import sys
sys.path.append('../../')

import keras2caffe

from keras.models import load_model
model = load_model('simple_CNN.81-0.96.hdf5')

keras2caffe.convert(model, 'deploy.prototxt', 'weights.caffemodel')

