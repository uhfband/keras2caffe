# Keras to Caffe model converter

This tool tested with Caffe 1.0, Keras 2.1.2 and TensorFlow 1.4.0

Working conversion examples:
- Inception V3
- Inception V4 (https://github.com/kentsommer/keras-inceptionV4)
- Xception V1
- SqueezeNet (https://github.com/rcmalli/keras-squeezenet)


Problem layers:
- ZeroPadding2D
- MaxPooling2D and AveragePooling2D with asymmetric padding


