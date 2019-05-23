import tensorflow as tf
from tensorflow import keras

class PairwiseCrossSimilarity(keras.layers.Layer):
  n_unique_scores = 0
  def __init__(self, relation_unit, k):
    '''
    Args:
      relation_unit: a RelationModule object used to compare pairwise features
      k: current level's number of co-objects
    '''
    super(PairwiseCrossSimilarity, self).__init__()
    self._relation_unit = relation_unit
    self._k = k
    assert(k >= 2)

  def build(self, input_shapes):
    hk = self._k//2
    self.conv2d = keras.layers.Conv2D(1, kernel_size=hk, strides=(hk,hk),
        padding='valid', bias_initializer='zeros', activation=None,
        kernel_initializer='ones', trainable=False)
    #self.conv2d.build()
    super(PairwiseCrossSimilarity, self).build(input_shapes)

  def call(self, inputs, training=None):
    '''
    Args:
      inputs: a list of [orig_fea,ind0,ind1] tensors. where
        orig_fea: is a tensor with shape [MBS*K, B, D] representing original features
        ind0,ind1: has shape [MBS*ncobj, num_top, cobj_size] representing current subtree
    '''
    assert(isinstance(inputs,list) and len(inputs) == 3), 'inputs must be a list of 3 tensors'

    orig_fea, ind0, ind1 = inputs
    def create_ids(ind):
      bag_size = orig_fea.shape[1]
      ids = ind + bag_size * tf.range(ind.shape[-1], dtype=tf.int32)
      return tf.reshape(ids, [ind.shape[0], -1])

    def unique_with_inverse(x):
      y, idx = tf.unique(x)
      num_segments = y.shape[0]
      num_elems = x.shape[0]
      return (y, idx, tf.math.unsorted_segment_max(tf.range(num_elems), idx, num_segments))

    ids0 = create_ids(ind0)
    ids1 = create_ids(ind1)

    def reduced_score(inp):
      solo_fea, ids0, ids1 = inp
      u_ids0, idx_map0, inverse0 = unique_with_inverse(ids0)
      u_ids1, idx_map1, inverse1 = unique_with_inverse(ids1)
      reduced_fea0 = tf.gather(solo_fea[0], u_ids0)
      reduced_fea1 = tf.gather(solo_fea[1], u_ids1)
      # [1, m', l', 1]
      reduced_scores = self._relation_unit([reduced_fea0[tf.newaxis,...],
          reduced_fea1[tf.newaxis,...]], training=training)
      nscores = reduced_scores.shape[1]*reduced_scores.shape[2]
      # [m', l', 1]
      reduced_scores = tf.reshape(reduced_scores, reduced_scores.shape[1:])
      # [m, l', 1]
      scores_0 = tf.gather(reduced_scores, idx_map0)
      # [m, l, 1]
      scores = tf.gather(scores_0, idx_map1, axis=1)
      return scores, nscores

    bs = ids0.shape[0]
    rsolo_fea = tf.reshape(orig_fea, [bs, 2, -1, orig_fea.shape[2]])
    scores, nscores = [],[]
    num = rsolo_fea.shape[0]
    assert(num is not None)
    for i in range(num):
      ret = reduced_score((rsolo_fea[i], ids0[i], ids1[i]))
      scores.append(ret[0])
      nscores.append(ret[1])
    scores = tf.stack(scores)
    nscores = tf.stack(nscores)
    PairwiseCrossSimilarity.n_unique_scores += tf.reduce_sum(nscores)
    # [N, m, l, 1]
    scores = self.conv2d(scores)
    # [N, m*l, 1]
    scores = tf.reshape(scores, [scores.shape[0],
                        scores.shape[1]*scores.shape[2],
                        scores.shape[3]])
    # [m, l, 2]
    pairs =  tf.stack(tf.meshgrid(tf.range(ind0.shape[1]),
                                  tf.range(ind1.shape[1]), indexing='ij'),
                      axis=-1)
    # [N, m, l, 2]
    pairs = tf.tile(pairs[tf.newaxis], [ind0.shape[0], 1, 1, 1])
    # [N, m*l, 2]
    pairs = tf.reshape(pairs, [pairs.shape[0],
                               pairs.shape[1]*pairs.shape[2],
                               pairs.shape[3]])
    return scores, pairs

