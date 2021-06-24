import tensorflow.keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input,Flatten,BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Add
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from bmi.customized_activation_function import CustomizedAct

# model architecture
def create_dense_model(n_cols,activation=None):
    activation = activation or CustomizedAct(10,60,0.1)
    # n_cols = len(train_array[0])
    # Input - Layer
    input_ = Input(shape=(n_cols))
    x = Embedding(11, 8)(input_)
    x = Flatten()(x)
    # block 1
    x = Dense(64, activation= LeakyReLU())(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    # block 2
    x = Dense(32, activation= LeakyReLU())(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    # Output- Layer
    x = Dense(1, activation=None)(x)
    output = activation(x)
    
    model = Model(input_, output)
    return model

#model.compile(optimizer = optimizers.Adam(lr=1e-1), loss='MSE')
def dense_block(input_data, units, drop_rate):
    x = Dense(units, activation= LeakyReLU())(input_data)
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x)
    #x = layers.Add()([x, input_data])
    return x



def create_deep_dense_model(input_cols, output_cols=None, activation=None):
    if activation == 'linear':
        activation = tf.keras.activations.linear
    else:
        activation = activation or CustomizedAct(10,60,0.1)
    # n_cols = len(train_array[0])
    # Input - Layer
    input_ = Input(shape=(input_cols))
    #x = Dropout(0.99)(input_)
    x = Embedding(11, 8)(input_)
    x = Flatten()(x)
    for i in range(5):
        x = dense_block(x, 64, 0.5)
    for i in range(5):
        x = dense_block(x, 32, 0.5)
    # Output- Layer
    output_cols= output_cols or 1
    x = Dense(output_cols, activation=None)(x)
    output = activation(x)
    model = Model(input_, output)
    return model

"""
def create_linear_dense_model(n_cols,activation=None):
    activation = activation or CustomizedAct(10,60,0.1)
    # n_cols = len(train_array[0])
    # Input - Layer
    input_ = Input(shape=(n_cols))
    #x = Dropout(0.99)(input_)
    x = Embedding(11, 8)(input_)
    x = Flatten()(x)
    for i in range(5):
        x = dense_block(x, 64, 0.5)
    for i in range(5):
        x = dense_block(x, 32, 0.5)
    # Output- Layer
    output = Dense(1, activation='linear')(x)
    model = Model(input_, output)
    return model
"""