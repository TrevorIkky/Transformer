import tensorflow as tf
import numpy as  np

from typing import Tuple
from attention import MultiHeadAttention
from posencoding import PositionalEncoding
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Embedding

tf.config.run_functions_eagerly(True)

@tf.function()
def EncoderLayer(x, mask, dim_model, heads, expansion=4, dropout_rate=0.1, training=False) -> tf.Tensor:

    a, o = MultiHeadAttention(x, x, x, dim_model, heads)
    x = Dropout(dropout_rate)(o, training=training)
    x = LayerNormalization()(x + o)

    y = Dense(dim_model * expansion, activation="relu")(x)
    y = Dense(dim_model)(y)
    y = Dropout(dropout_rate)(y, training=training)
    y = LayerNormalization()(x + y)
    return y

@tf.function()
def Encoder(x, vocab_size, dim_model, heads, dropout_rate=0.1, layers=4, training=False) -> tf.Tensor:
    S = x.shape
    x =  Embedding(vocab_size, dim_model)(x)
    p = PositionalEncoding(vocab_size, dim_model)
    x += p[:, :S[-1], :]

    for _ in range(layers):
        x = EncoderLayer(x, None, dim_model, heads, training=training)
    x = Dropout(dropout_rate)(x, training=training)
    return x

if __name__ == "__main__":
    i = tf.random.uniform((20, 30), dtype=tf.int64, minval=0, maxval=20)
    print(f'Input shape: {i.shape}')
    N = Encoder(i, 200, 512, 8, training=True)
    print(f'{N.shape}')
