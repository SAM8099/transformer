import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras import Sequential

class FeedForwardNetwork(Layer):
    def __init__(self, d_model=128, dff=512, rate=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.ffn = Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model),
            Dropout(rate)
        ])

    def call(self, x, training=False):
        return self.ffn(x, training=training)