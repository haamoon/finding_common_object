import tensorflow as tf
from tensorflow import keras

#@tf.function
def create_ids(ind, bag_size):
  ids = ind + bag_size * tf.range(ind.shape[-1], dtype=tf.int32)
  return tf.reshape(ids, [ind.shape[0], -1])

#@tf.function
def unique_with_inverse(x):
  y, idx = tf.unique(x)
  num_segments = tf.shape(y)[0]
  num_elems = tf.shape(x)[0]
  return (y, idx, tf.math.unsorted_segment_max(tf.range(num_elems),
                                               idx, num_segments))
#@tf.function
def reduced_score(inp, relation_unit, training):
  solo_fea, ids0, ids1 = inp
  u_ids0, idx_map0, inverse0 = unique_with_inverse(ids0)
  u_ids1, idx_map1, inverse1 = unique_with_inverse(ids1)
  reduced_fea0 = tf.gather(solo_fea[0], u_ids0)
  reduced_fea1 = tf.gather(solo_fea[1], u_ids1)
  # [1, m', l', 1]
  reduced_scores = relation_unit([reduced_fea0[tf.newaxis,...],
              reduced_fea1[tf.newaxis,...]], training=training)
  reduced_scores_shape = tf.shape(reduced_scores)
  nscores = reduced_scores_shape[1]*reduced_scores_shape[2]
  # [m', l', 1]
  reduced_scores = tf.reshape(reduced_scores, reduced_scores_shape[1:])
  # [m, l', 1]
  scores_0 = tf.gather(reduced_scores, idx_map0)
  # [m, l, 1]
  scores = tf.gather(scores_0, idx_map1, axis=1)
  return scores, nscores

#@tf.function
def sum_kxk_patches(inp, k):
  '''
    inp: A tensor with shape [m, h, w, c]
    k: A scalar tensor
  '''
  # [m, h, w, c]
  inp_shape = tf.shape(inp)
  hk = tf.cast(inp_shape[1]/k, dtype=tf.int32)
  wk = tf.cast(inp_shape[2]/k, dtype=tf.int32)
  k = tf.cast(k, dtype=tf.int32)

  # Reshape to [m, h/k, k, w/k, k, c]
  inp = tf.reshape(inp, [inp_shape[0], hk, k, wk, k, inp_shape[3]])
  return tf.reduce_sum(inp, [2, 4])

#@tf.function
def get_pairs(N, m, l):
    # [m, l, 2]
    pairs =  tf.stack(tf.meshgrid(tf.range(m),
                                  tf.range(l), indexing='ij'),
                                  axis=-1)
    # [N, m, l, 2]
    pairs = tf.tile(pairs[tf.newaxis], [N, 1, 1, 1])
    # [N, m*l, 2]
    pairs_shape = tf.shape(pairs)
    pairs = tf.reshape(pairs, [pairs_shape[0],
                               pairs_shape[1]*pairs_shape[2],
                               pairs_shape[3]])
    return pairs

class PairwiseCrossSimilarity(keras.layers.Layer):
  def __init__(self, relation_unit):
    '''
    Args:
      relation_unit: a RelationModule object used to compare pairwise features
      k: current level's number of co-objects
    '''
    super(PairwiseCrossSimilarity, self).__init__()
    self._relation_unit = relation_unit

  #@tf.function
  def call(self, inputs, training=None):
    '''
    Args:
      inputs: a list of [orig_fea,ind0,ind1] tensors. where
        orig_fea: is a tensor with shape [MBS*K, B, D] representing original features
        ind0,ind1: has shape [MBS*ncobj, num_top, cobj_size] representing current subtree
        k: k is the size of output cobj_size
    '''
    assert(isinstance(inputs,list) and len(inputs) == 4
                  ), 'inputs must be a list of 3 tensors'

    orig_fea, ind0, ind1, k = inputs
    bag_size = tf.shape(orig_fea)[1]

    ids0 = create_ids(ind0, bag_size)
    ids1 = create_ids(ind1, bag_size)

    num = tf.shape(ids0)[0]
    rsolo_fea = tf.reshape(orig_fea, [num, 2, -1, tf.shape(orig_fea)[2]])
    #from IPython import embed;embed()
    scores = tf.TensorArray(tf.float32, size=num)
    nscores = tf.TensorArray(tf.int32, size=num)
    for i in tf.range(num):
      ret = reduced_score((rsolo_fea[i], ids0[i], ids1[i]),
                           self._relation_unit, training)
      scores = scores.write(i, ret[0])
      nscores = nscores.write(i, ret[1])
    scores = scores.stack()
    nscores = nscores.stack()
    #PairwiseCrossSimilarity.n_unique_scores += tf.reduce_sum(nscores)
    m, l = tf.shape(ind0)[1], tf.shape(ind1)[1]

    # [N, m, l, 1]
    scores = sum_kxk_patches(scores, k//2)

    # [N, m*l, 1]
    scores = tf.reshape(scores, [num, m*l, 1])

    pairs = get_pairs(num, m, l)
    return scores, pairs
