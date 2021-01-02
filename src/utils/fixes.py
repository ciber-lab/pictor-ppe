import tensorflow as tf


def fix_tf_gpu():
    '''
    Fix for the following error message:
    UnknownError: Failed to get convolution algorithm. 
    This is probably because cuDNN failed to initialize...

    More:
    https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
    '''

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass