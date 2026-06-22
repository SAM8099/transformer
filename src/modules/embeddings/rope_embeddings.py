import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

class ROPEEmbedding(Layer):
    def __init__(self, max_len=128, d_model=128):
        super(ROPEEmbedding, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        