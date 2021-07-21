import tensorflow as tf

from typing import Tuple
from encoder import Encoder
from posencoding import PositionalEncoding
from attention import MultiHeadAttention
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, Dropout

tf.config.run_functions_eagerly(True)

@tf.function()
def DecoderLayer(dO, x, mask, dim_model, heads, expansion=4, dropout_rate=0.1, training=False) -> Tuple:
    v, mA = MultiHeadAttention(x, x, x, dim_model, heads)
    v = Dropout(dropout_rate)(v, training=training)
    v = LayerNormalization()(v + x)

    y, a = MultiHeadAttention(dO, dO, v, dim_model, heads)
    y = Dropout(dropout_rate)(y, training=training)
    y = LayerNormalization()(v + y)

    z = Dense(dim_model * expansion, activation="relu")(y)
    z = Dense(dim_model)(z)
    z = Dropout(dropout_rate)(z, training=training)
    z = LayerNormalization()(y + z)

    return z, mA, a

@tf.function()
def Decoder(dO, x, vocab_size, dim_model, heads, layers=4, dropout_rate=0.1, training=False) -> tf.Tensor:
        S = x.shape
        x = Embedding(vocab_size, dim_model)(x)
        p = PositionalEncoding(vocab_size, dim_model)
        x += p[:, :S[-1], :]

        for _ in range(layers):
            x, _, _ = DecoderLayer(dO, x, None, dim_model, heads, training=training)
        x = Dropout(dropout_rate)(x, training=training)
        return x

if __name__ == "__main__":
    V = 200
    E = 512
    i = tf.random.uniform((20, 30), dtype=tf.float32, minval=0, maxval=30)
    N = Encoder(i, V, E, 8, training=True)
    print(f'N output {N.shape}')
    D = Decoder(N, i, V, E, 8, training=True)
    print(f'D output: {D.shape}')
