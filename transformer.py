import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model

tf.config.run_functions_eagerly(True)

@tf.function()
def Transformer(I, T, vocab_size, dim_model, heads, layers, training=False ) -> Model:
    I = Input(shape=I.shape)
    print(f'I shape = {I.shape}')
    X = Encoder(I, vocab_size, dim_model, heads, layers=4, dropout_rate=0.1, training=training)
    X = Decoder(X, T, vocab_size, dim_model, heads, layers=4, dropout_rate=0.1, training=training)
    X = Dense(vocab_size, activation="softmax")(X)
    M = Model(inputs=I , outputs=X)
    return M


if __name__ == "__main__":
    V = 1000
    E = 512
    H = 8
    I = tf.random.uniform((20,30), dtype=tf.float32, minval=0, maxval=40)
    T = tf.random.uniform((20, 60), dtype=tf.float32, minval=0, maxval=50)
    X = Transformer(I, T, V, E, H, 8, training=True)
    print(f'X shape: {X.summary()}')
