import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dropout

from src.modules.attention.self_attention import SelfAttention
from src.modules.feedforward.feedforward_network import FeedForwardNetwork

class Encoder(Layer):
    def __init__(self, d_model=128, num_heads=4, dff=512, rate=0.1):
        super(Encoder, self).__init__()
        
        # Core blocks (Using your custom SelfAttention and FeedForward classes)
        self.mha = SelfAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FeedForwardNetwork(d_model=d_model, dff=dff, rate=rate)
        
        # Add & Norm components
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(rate)

    def call(self, x, mask=None, training=False):
        
        attn_output = self.mha(v=x, k=x, q=x, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(attn_output + x)  # Skip connection

        ffn_output = self.ffn(out1, training=training)
        out2 = self.layernorm2(ffn_output + out1)  # Skip connection
        
        return out2