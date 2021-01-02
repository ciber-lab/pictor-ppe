import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2

'''===================================================================================================='''
'''BLOCKS'''

'''Convolutional Block'''
def yolo_ConvBlock (input_tensor, num_filters, filter_size, strides = (1,1) ):
    padding = 'valid' if strides == (2,2) else 'same'

    ### Layers
    x = Conv2D( num_filters, filter_size, strides, padding, use_bias=False, kernel_regularizer=l2(5e-4) ) (input_tensor)
    x = BatchNormalization() (x)
    x = LeakyReLU(alpha=0.1) (x)
    
    return x

'''Residual Block'''
def yolo_ResidualBlocks (input_tensor, num_filters, num_blocks ):
    
    ### Layers
    x = ZeroPadding2D( ((1,0),(1,0)) ) (input_tensor) # left & top padding
    x = yolo_ConvBlock ( x, num_filters, filter_size=(3,3), strides = (2,2) )
    
    for _ in range( num_blocks ):
        y = yolo_ConvBlock ( x, num_filters//2, filter_size=(1,1), strides = (1,1) )
        y = yolo_ConvBlock ( y, num_filters   , filter_size=(3,3), strides = (1,1) )
        x = Add() ([x, y])
    
    return x

'''Output Block'''
def yolo_OutputBlock (x, num_filters, out_filters ):
    
    ### Layers
    x = yolo_ConvBlock ( x, 1*num_filters, filter_size=(1,1), strides = (1,1) )
    x = yolo_ConvBlock ( x, 2*num_filters, filter_size=(3,3), strides = (1,1) )
    x = yolo_ConvBlock ( x, 1*num_filters, filter_size=(1,1), strides = (1,1) )
    x = yolo_ConvBlock ( x, 2*num_filters, filter_size=(3,3), strides = (1,1) )
    x = yolo_ConvBlock ( x, 1*num_filters, filter_size=(1,1), strides = (1,1) )
    
    y = yolo_ConvBlock ( x, 2*num_filters, filter_size=(3,3), strides = (1,1) )
    y = Conv2D ( filters=out_filters, kernel_size=(1,1), strides=(1,1), 
                padding='same', use_bias=True, kernel_regularizer=l2(5e-4) ) (y)
    
    return x, y

'''===================================================================================================='''
'''COMPLETE MODEL'''

def yolo_body (input_tensor, num_out_filters):
    '''
    Input: 
        input_tensor   = Input( shape=( *input_shape, 3 ) )
        num_out_filter = ( num_anchors // 3 ) * ( 5 + num_classes )
    Output:
        complete YOLO-v3 model
    '''

    # 1st Conv block
    x = yolo_ConvBlock( input_tensor, num_filters=32, filter_size=(3,3), strides=(1,1) )

    # 5 Resblocks
    x = yolo_ResidualBlocks ( x, num_filters=  64, num_blocks=1 )
    x = yolo_ResidualBlocks ( x, num_filters= 128, num_blocks=2 )
    x = yolo_ResidualBlocks ( x, num_filters= 256, num_blocks=8 )
    x = yolo_ResidualBlocks ( x, num_filters= 512, num_blocks=8 )
    x = yolo_ResidualBlocks ( x, num_filters=1024, num_blocks=4 )

    darknet = Model( input_tensor, x ) # will use it just in a moment

    # 1st output block
    x, y1 = yolo_OutputBlock( x, num_filters= 512, out_filters=num_out_filters )

    # 2nd output block
    x = yolo_ConvBlock( x, num_filters=256, filter_size=(1,1), strides=(1,1) )
    x = UpSampling2D(2) (x)
    x = Concatenate() ( [x, darknet.layers[152].output] )
    x, y2 = yolo_OutputBlock( x, num_filters= 256, out_filters=num_out_filters )

    # 3rd output block
    x = yolo_ConvBlock( x, num_filters=128, filter_size=(1,1), strides=(1,1) )
    x = UpSampling2D(2) (x)
    x = Concatenate() ( [x, darknet.layers[92].output] )
    x, y3 = yolo_OutputBlock( x, num_filters= 128, out_filters=num_out_filters )

    # Final model
    model = Model( input_tensor, [y1, y2, y3] )
    
    return model
