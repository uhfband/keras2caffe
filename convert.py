import sys
# sys.path.append('../../')
import os
from datetime import datetime
import keras2caffe

from keras.models import load_model, model_from_json

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception('ERROR: No input model.')
    model_file = sys.argv[1]
    if not os.path.isfile(model_file):
        raise Exception('ERROR: model file is not exist!')

    if len(sys.argv) < 3:
        raise Exception('ERROR: No input weights.')
    w_file = sys.argv[2]
    if not os.path.isfile(model_file):
        raise Exception('ERROR: w file is not exist!')

    OUTPUT_DIR = datetime.now().strftime("output_%y%m%d_%H%M%S/")
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model = model_from_json(open(sys.argv[1], 'r').read()) if sys.argv[1][-5:] == '.json'  else load_model(sys.argv[1])
    model.load_weights(sys.argv[2])

    keras2caffe.convert(model, OUTPUT_DIR + 'deploy.prototxt', OUTPUT_DIR + 'model.caffemodel')

