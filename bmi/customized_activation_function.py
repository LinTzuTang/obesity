import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils

# customized activation function
class CustomizedAct(Layer):
    def __init__(self,min_value,max_value,alpha):
        super().__init__()
        self._min_value = min_value
        self._max_value = max_value
        self._alpha = alpha

    def get_config(self):
        base_config = super().get_config()
        base_config.update({'alpha':self._alpha,'min':self._min_value,'max':self._max_value})
        return base_config
        

    def call(self, x, training=True):
        alpha = self._alpha if training else 0
        x_1 = tf.nn.leaky_relu(x-self._min_value,alpha)+self._min_value
        x_2 = -tf.nn.leaky_relu(-x+self._max_value,alpha)+self._max_value
        return x_1 + x_2 - x
    
    
    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape