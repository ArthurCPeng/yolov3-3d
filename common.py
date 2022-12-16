import tensorflow as tf
from tensorflow.keras import layers

def conv_block(input_data, kernel, filters, name,
               trainable = True, downsample = False,
               activate = True, bn = True):
    if downsample:
        strides = (2, 2, 2)
    else:
        strides = (1, 1, 1)

    conv = layers.Conv3D(filters, kernel,
                         strides=strides, padding = 'same',
                         name = name + '_conv')(input_data)

    if bn:
        conv = layers.BatchNormalization(name = name + '_batchnorm')(conv)

    if activate:
        conv = layers.LeakyReLU(alpha=0.1, name = name + '_leakyrelu')(conv)
    return conv

def conv_series(input_data, kernels, filters, name, trainable=True):
    assert len(kernels) == len(filters)
    for i in range(len(kernels)):
        input_data = conv_block(input_data, kernels[i], filters[i], trainable=trainable,
                                name = name+'_convblock{}'.format(i+1))
    return input_data
    

def res_block(input_data, filters1, filters2, name, trainable=True):
    shortcut = input_data

    input_data = conv_block(input_data, (1,1,1), filters1, name=name+'_conv1')
    input_data = conv_block(input_data, (3,3,3), filters2, name=name+'_conv2')

    return input_data + shortcut
    
    
def darknet53(input_data, trainable=True):
    input_data = conv_block(input_data, (3,3,3), factor * 2, trainable=trainable, name='conv1')
    input_data = conv_block(input_data, (3,3,3), factor * 4, trainable=trainable, name='conv2',
                            downsample = True)

    for i in range(1):
        input_data = res_block(input_data, factor*2, factor*4,
                               name = 'res_stage1_{}'.format(i+1), trainable=trainable)

    input_data = conv_block(input_data, (3,3,3), factor * 8, trainable=trainable, name='conv3',
                            downsample = True)
    
    for i in range(2):
        input_data = res_block(input_data, factor*4, factor*8,
                               name = 'res_stage2_{}'.format(i+1), trainable=trainable)

    input_data = conv_block(input_data, (3,3,3), factor * 16, trainable=trainable, name='conv4',
                            downsample = True)

    for i in range(8):
        input_data = res_block(input_data, factor*8, factor*16,
                               name = 'res_stage3_{}'.format(i+1), trainable=trainable)
    route1 = input_data
    input_data = conv_block(input_data, (3,3,3), factor * 32, trainable=trainable, name='conv5',
                            downsample = True)

    for i in range(8):
        input_data = res_block(input_data, factor*16, factor*32,
                               name = 'res_stage4_{}'.format(i+1), trainable=trainable)
    
    route2 = input_data
    input_data = conv_block(input_data, (3,3,3), factor * 64, trainable=trainable, name='conv6',
                            downsample = True)
    for i in range(4):
        input_data = res_block(input_data, factor*32, factor*64,
                               name = 'res_stage5_{}'.format(i+1), trainable=trainable)

    return route1, route2, input_data

def upsample_block(input_data, name, method='deconv'):
    input_shape = input_data.shape
    if method == 'resize':
        '''
        output = tf.compat.v1.image.resize_nearest_neighbor(
            input_data, (input_shape[1] * 2, input_shape[2] * 2))
        '''
        output = layers.Upsample3D(size=(2,2,2))(input_data)
        
    if method == 'deconv':
        output = layers.Conv3DTranspose(input_shape[-1], kernel_size = 2,
                                        strides=(2,2,2), name = name + '_deconv')(input_data)
    return output
