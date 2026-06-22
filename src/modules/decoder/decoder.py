import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dropout

from src.modules.attention.self_attention import SelfAttention
from src.modules.feedforward.feedforward_network import FeedForwardNetwork

class Decoder(Layer):
    def __init__(self, d_model=128, num_heads=4, dff=512, rate=0.1):
        super().__init__()
        
        # Block 1: Masked Self-Attention over target tokens
        self.mha1 = SelfAttention(d_model=d_model, num_heads=num_heads)
        
        # Block 2: Cross-Attention (Connects Decoder to Encoder output)
        self.mha2 = SelfAttention(d_model=d_model, num_heads=num_heads)
        
        # Block 3: Feed-Forward Network
        self.ffn = FeedForwardNetwork(d_model=d_model, dff=dff, rate=rate)
        
        # Add & Norm layers for each block
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, enc_output, look_ahead_mask=None, padding_mask=None, training=False):
        # 1. Masked Self-Attention (Q, K, V all come from target input 'x')
        attn1 = self.mha1(v=x, k=x, q=x, mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)  # Skip connection 1
        
        # 2. Cross-Attention
        # Q comes from the decoder sequence (out1)
        # K and V come from the encoder's final output (enc_output)
        attn2 = self.mha2(v=enc_output, k=enc_output, q=out1, mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # Skip connection 2
        
        # 3. Feed-Forward Network
        ffn_output = self.ffn(out2, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # Skip connection 3
        
        return out3