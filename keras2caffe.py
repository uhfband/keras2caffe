import caffe
from caffe import layers as L, params as P

import numpy as np


def convert(keras_model, caffe_net_file, caffe_params_file):
    
    caffe_net = caffe.NetSpec()
    
    net_params = dict()
    
    outputs=dict()
    shape=()
    
    input_str = ''
    
    for layer in keras_model.layers:
        name = layer.name
        layer_type = type(layer).__name__
        
        config = layer.get_config()
        
        blobs = layer.get_weights()
        blobs_num = len(blobs)
        
        
        if type(layer.output)==list:
            raise Exception('Layers with multiply outputs are not supported')
        else: 
            top=layer.output.name
        
        if type(layer.input)!=list:
            bottom = layer.input.name
            
        
        if layer_type=='InputLayer':
            name = 'data'
            caffe_net[name] = L.Layer()
            input_shape=config['batch_input_shape']
            input_str = 'input: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}'.format('"' + name + '"',
                1, input_shape[3], input_shape[1], input_shape[2])
            
        
        elif layer_type=='Conv2D' or layer_type=='Convolution2D':
            strides = config['strides']
            kernel_size = config['kernel_size']
            filters = config['filters']
            padding = config['padding']
            use_bias = config['use_bias']
            
            blobs[0] = np.array(blobs[0]).transpose(3,2,0,1)
            
            param = dict(bias_term = use_bias)
            
            if kernel_size[0]==kernel_size[1]:
                pad=0
                if padding=='same':
                    pad=kernel_size[0]/2
                caffe_net[name] = L.Convolution(caffe_net[outputs[bottom]], num_output=filters, 
                    kernel_size=kernel_size[0], pad=pad, stride=strides[0], convolution_param=param)
            else:
                pad_h = pad_w = 0
                if padding=='same':
                    pad_h = kernel_size[0]/2
                    pad_w = kernel_size[1]/2
                caffe_net[name] = L.Convolution(caffe_net[outputs[bottom]], num_output=filters, 
                    kernel_h=kernel_size[0], kernel_w=kernel_size[1], pad_h=pad_h, pad_w=pad_w, stride=strides[0], convolution_param=param)
            
            net_params[name] = blobs
            
        elif layer_type=='BatchNormalization':
            caffe_net[name] = L.BatchNorm(caffe_net[outputs[bottom]], in_place=True)
            
            variance = np.array(blobs[-1])
            mean = np.array(blobs[-2])
            
            param = dict()
            
            if config['scale']:
                gamma = np.array(blobs[-3])
            else:
                gamma = np.ones(mean.shape, dtype=np.float32)
            
            if config['center']:
                beta = np.array(blobs[0])
                param['bias_term']=True
            else:
                beta = np.zeros(mean.shape, dtype=np.float32)
                param['bias_term']=False
            
            net_params[name] = (mean, variance, np.array(1.0)) 
            
            name_s = name+'s'
            
            caffe_net[name_s] = L.Scale(caffe_net[name], in_place=True, scale_param=param)
            net_params[name_s] = (gamma, beta)
            
        elif layer_type=='Dense':
            caffe_net[name] = L.InnerProduct(caffe_net[outputs[bottom]], num_output=config['units'])
            
            if config['use_bias']:
                net_params[name] = (np.array(blobs[0]).transpose(1, 0), np.array(blobs[1]))
            else:
                net_params[name] = (blobs[0])
        
        elif layer_type=='Activation':
            if config['activation']!='relu':
                raise Exception('Unsupported activation')
            caffe_net[name] = L.ReLU(caffe_net[outputs[bottom]], in_place=True)
            
        elif layer_type=='Concatenate':
            layers = []
            for i in layer.input:
                layers.append(caffe_net[outputs[i.name]])
            caffe_net[name] = L.Concat(*layers, axis=1)
        
        elif layer_type=='Add':
            layers = []
            for i in layer.input:
                layers.append(caffe_net[outputs[i.name]])
            caffe_net[name] = L.Eltwise(*layers)
        
        elif layer_type=='Flatten':
            caffe_net[name] = L.Flatten(caffe_net[outputs[bottom]])
        
        elif layer_type=='MaxPooling2D' or layer_type=='AveragePooling2D':
            if layer_type=='MaxPooling2D':
                pool = P.Pooling.MAX
            else:
                pool = P.Pooling.AVE
                
            pool_size = config['pool_size']
            strides  = config['strides']
            padding = config['padding']
            
            if pool_size[0]!=pool_size[1]:
                raise Exception('Unsupported pool_size')
                    
            if strides[0]!=strides[1]:
                raise Exception('Unsupported strides')
            
            pad=0
            if padding=='same':
                pad=pool_size[0]/2
                
            caffe_net[name] = L.Pooling(caffe_net[outputs[bottom]], kernel_size=pool_size[0], 
                stride=strides[0], pad=pad, pool=pool)
        
        #TODO
        
        elif layer_type=='GlobalAveragePooling2D':
            caffe_net[name] = L.Pooling(caffe_net[outputs[bottom]], kernel_size=8, stride=8, pad=0, pool=P.Pooling.AVE)
        
        elif layer_type=='ZeroPadding2D':
            padding=config['padding']
            caffe_net[name] = L.Convolution(caffe_net[outputs[bottom]], num_output=3, kernel_size=1, stride=1,
                pad_h=padding[0][0], pad_w=padding[1][0], convolution_param=dict(bias_term = False))
            net_params[name] = np.ones((3,3,1,1))
        
        else:
            raise Exception('Unsupported layer type: '+layer_type)
            
        outputs[top]=name
        
    
    #replace empty layer with input blob
    net_proto = input_str + '\n' + 'layer {' + 'layer {'.join(str(caffe_net.to_proto()).split('layer {')[2:])
    
    f = open(caffe_net_file, 'w') 
    f.write(net_proto)
    f.close()
    
    caffe_model = caffe.Net(caffe_net_file, caffe.TEST)
    
    for layer in caffe_model.params.keys():
        for n in range(0, len(caffe_model.params[layer])):
            caffe_model.params[layer][n].data[...] = net_params[layer][n]

    caffe_model.save(caffe_params_file)
