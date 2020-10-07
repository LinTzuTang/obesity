# model architecture
import tensorflow.keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input,Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import Model
from tensorflow.keras import optimizers

def dense_model(train_data):
    #get number of columns in training data
    n_cols = train_data.shape[1]
    # Input - Layer
    input_ = Input(shape=(n_cols))
    x = Embedding(11, 8)(input_)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    # Output- Layer
    output = Dense(1, activation = "sigmoid")(x)
    
    model = Model(input_, output)
    model.compile(optimizer = optimizers.Adam(lr=5*1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model