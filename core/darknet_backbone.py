import core.common as common
import tensorflow as tf

def darknet53(input_data, trainable):
    
    with tf.variable_scope('darknet'):
        
        output_data = common.convolutional(input_data, filter_shape=(3,3,32,32), 
                                           trainable=trainable, name='conv1')
        
        output_data = common.convolutional(output_data, filter_shape=(3,3,32,64),
                                           trainable=trainable, downsample=True, name='conv2')
        
        for i in range(1):
            output_data = common.residual_block(output_data, 64, 32, 64,
                                                trainable=trainable, name='residual%d'%(i+1))
            
        output_data = common.convolutional(output_data, filter_shape=(3,3,64,128),
                                           trainable=trainable, downsample=True, name='conv5')
        
        for i in range(2):
            
            output_data = common.residual_block(output_data, 128, 64, 128,
                                                trainable=trainable, name='residual%d'%(i+2))
            
        output_data = common.convolutional(output_data, filter_shape=(3,3,128,256),
                                           trainable=trainable, downsample=True, name='conv10')
        
        for i in range(8):
            
            output_data = common.residual_block(output_data, 256, 128, 256,
                                                trainable=trainable, name='residual%d'%(i+4))
            
        route_1 = output_data
        
        output_data = common.convolutional(output_data, filter_shape=(3,3,256,512),
                                           trainable=trainable, downsample=True, name='conv27')
        
        for i in range(8):
            
            output_data = common.residual_block(output_data, 512, 256, 512,
                                                trainable=trainable, name='residual%d'%(i+12))
            
        route_2=output_data
        
        output_data = common.convolutional(output_data, filter_shape=(3,3,512,1024),
                                           trainable=trainable, downsample=True, name='conv44')
        
        for i in range(4):
            output_data = common.residual_block(output_data, 1024, 512, 1024,
                                                trainable=trainable, name='residual%d'%(i+20))
            
    return route_1, route_2, output_data
        
            