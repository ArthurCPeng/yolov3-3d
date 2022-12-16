import tensorflow as tf
from tensorflow.keras import layers
from common import *
from config import cfg

    
def darknet53(input_data, trainable=True):

    factor = cfg.YOLO.FACTOR
    
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
