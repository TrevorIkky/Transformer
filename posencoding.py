import tensorflow as tf
import numpy as np

def PositionalEncoding(max_seq_len, embedding_dim) -> tf.Tensor:
    if embedding_dim % 2 == 1 : embedding_dim += 1
    gX, gY = np.meshgrid(np.arange(max_seq_len), np.arange(embedding_dim // 2))
    E = np.empty((1, max_seq_len, embedding_dim))
    E[0, :, ::2] = np.sin(gX / 10000**(2 * gY / embedding_dim)).T
    E[0, :, 1::2] = np.cos(gX / 10000**(2 * gY / embedding_dim)).T
    pE = tf.cast(E, dtype=tf.float32)
    return pE
