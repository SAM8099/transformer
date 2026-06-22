import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Embedding

class PositionalEmbedding(Layer):
    def __init__(self, vocab_size, d_model=128, max_seq_len=256, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.token_embedding = Embedding(input_dim=vocab_size, output_dim=d_model)
        
        pos = np.arange(max_seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) 
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        self.pos_encoding = tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        
        x = self.token_embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        return x + self.pos_encoding[:, :seq_len, :]