import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import LayerNormalization

class HParams:
    def __init__(self, n_vocab, n_ctx, n_embd, n_head, n_layer):
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

    def override_from_dict(self, hparam_dict):
        for key, value in hparam_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown hparam: {key}")

def default_hparams():
    return HParams(
        n_vocab=50257,
        n_ctx=1024,
        n_embd=1600,
        n_head=25,
        n_layer=48,
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, axis=-1, epsilon=1e-5):
    return LayerNormalization(axis=axis, epsilon=epsilon)(x)

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, nf, *, w_init_stdev=0.02):
    nx = x.shape[-1]
    w = tf.Variable(initial_value=tf.random.normal([1, nx, nf], stddev=w_init_stdev), trainable=True)
    b = tf.Variable(initial_value=tf.zeros([nf]), trainable=True)
    c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
    return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)

def attn(x, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.math.rsqrt(tf.cast(v.shape[-1], w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    c = conv1d(x, n_state*3)
    q, k, v = map(split_heads, tf.split(c, 3, axis=2))
    present = tf.stack([k, v], axis=1)
    if past is not None:
        pk, pv = tf.unstack(past, axis=1)
        k = tf.concat([pk, k], axis=-2)
        v = tf.concat([pv, v], axis=-2)
    a = multihead_attn(q, k, v)
    a = merge_heads(a)
    a = conv1d(a, n_state)
    return a, present

def mlp(x, n_state, *, hparams):
    nx = x.shape[-1]
    h = gelu(conv1d(x, n_state))
    h2 = conv1d(h, nx)
    return h2

def block(x, *, past, hparams):
    nx = x.shape[-1]
    a, present = attn(norm(x), nx, past=past, hparams=hparams)
    x = x + a
    m = mlp(norm(x), nx*4, hparams=hparams)
    x = x + m
    return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)

class Transformer(Layer):
    def __init__(self, hparams, name='transformer', **kwargs):
        super(Transformer, self).__init__(name=name, **kwargs)
        self.hparams = hparams
        self.wpe = self.add_weight(shape=(hparams.n_ctx, hparams.n_embd), initializer=tf.random_normal_initializer(stddev=0.01))
        self.wte = self.add_weight(shape=(hparams.n_vocab, hparams.n_embd), initializer=tf.random_normal_initializer(stddev=0.02))

    def call(self, X, past=None):
        results = {}
        batch, sequence = shape_list(X)

        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(self.wte, X) + tf.gather(self.wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * self.hparams.n_layer
        assert len(pasts) == self.hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, past=past, hparams=self.hparams)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h)

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, self.hparams.n_embd])
        logits = tf.matmul(h_flat, self.wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, self.hparams.n_vocab])
        results['logits'] = logits
        return results
