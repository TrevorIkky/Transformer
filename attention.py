import tensorflow as tf

from typing import Tuple
from math import sqrt
from  tensorflow.keras.layers import Dense

tf.config.run_functions_eagerly(True)

@tf.function()
def MultiHeadAttention(keys, values, queries, dim_model, heads, mask=None) -> Tuple:
    assert dim_model % heads == 0, "Dim model %\ heads != 0"

    #print(f'Keys shape {keys.shape}')

    K = keys.shape
    V = values.shape
    Q = queries.shape

    k = Dense(dim_model)(keys)
    q = Dense(dim_model)(queries)
    v = Dense(dim_model)(values)

    k  = tf.reshape(k, [K[-3], K[-2], heads, dim_model // heads])
    v = tf.reshape(v, [V[-3], V[-2], heads, dim_model // heads])
    q = tf.reshape(q, [Q[-3], Q[-2], heads, dim_model // heads])

    a = tf.einsum("nqhd, nkhd -> nhqk", q, k)

    if mask is not None:
        a  += (mask * -1e9)

    a = tf.nn.softmax(a * (1 / sqrt(dim_model // heads)))
    a = tf.einsum("nhqk, nkhd -> nhqd", a, v)
    a = tf.reshape(a, [Q[-3], Q[-2], dim_model])

    x = Dense(dim_model)(a)

    return  a, x
