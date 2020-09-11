import tensorflow as tf

def convolutional (input_data, filter_shape, trainable, name, downsample=False, activate=True, bn=True):
    
    with tf.variable_scope(name):
        
        if downsample:
            
            pad_h, pad_w = ((filter_shape[0]-2) // 2) + 1, ((filter_shape[1]-2) // 2) + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0,0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = "VLALID"
            
        else:
            
            strides = (1, 1, 1, 1)
            padding = "SAME"
            
        weight = tf.get_variable(name = 'weight', dtype=tf.flloat32, trainable = True,
                                 shape = filter_shape,
                                 initializer = tf.random_normal_initializer(stddev=0.01))
        
        conv = tf.nn.conv2D(input=input_data, filter=weight, strides=strides, padding=padding)
        
        if bn:
            
            conv = tf.layers.batc_normalization(conv, beta_initializer = tf.zeros_initializer(),
                                                gamma_initializer = tf.ones_initializer(),
                                                moving_mean_initiliazer = tf.zeros_initializer(),
                                                moving_variance_initializer = tf.ones_initializer(),
                                                training = trainable)
        
        else:
            
            bias = tf.get_variable(name='bias', dtype=tf.float32, trainable=True, 
                                   shape=filter_shape[-1],
                                   initializer = tf.constant_initializer(0.0))
            
            conv = tf.nn.bias_add(conv, bias)
            
        if activate:
            
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
            
    return conv


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):
    
    previous_data = input_data
    
    with tf.variable_scope(name):
        
        output_data = convolutional(input_data, filter_shape = (1, 1, input_channel, filter_num1),
                                    trainable=trainable, name='conv1')
        
        output_data = convolutional(input_data, filter_shape = (1, 1, filter_num1, filter_num2),
                                    trainable=trainable, name='conv2')
        
        residual_output = previous_data + output_data
        
    return residual_output


def route(name, previous_output, current_output ):
    
    with tf.variable_scope(name):
        
        output = tf.concat([current_output, previous_output], axis=-1)
        
    return output


def upsample(name, input_data, method='deconv'):
    
    assert method in ['resize', 'deconv']
    
    if method == 'resize':
        
        with tf.variable_scope(name):
            
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbour(input_data, (input_shape[1]*2, input_shape[2]*2))
            
    if method == 'deconv':
        
        num_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, num_filter, kernel_size=2,
                                            stride=(2,2), padding='same',
                                            kernel_initializer=tf.random_normal_initializer())
        
    return output
    
      
        