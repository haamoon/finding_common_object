import tensorflow as tf
from tensorflow import keras
import numpy as np
import model.util as util

class UnaryModule(keras.layers.Layer):
  def __init__(self, relation_unit, fast_similarity=None,
                     mode='SOFTMAX', topk=None):
    """
    Args:
      relation_unit: used to compute similarity between positive and
        negative bags.
      fast_similarity: here it is used to select top similar instances
        in a fast manner.
      mode: determines how to compute the similarity between a positive
        image and all negative images
      topk: number of top elements from the negative bag to compute similarity.
        it can be None when we are not selecting topk elements.
    """
    super(UnaryModule, self).__init__()
    if mode not in ['MAX', 'MEAN', 'SOFTMAX']:
      raise ValueError('mode {} is not valid.'.format(mode))
    self._relation_unit = relation_unit
    self._fast_similarity = fast_similarity
    self._mode = mode
    self._topk = topk

  def build(self, input_shapes):
    if self._mode == 'SOFTMAX':
      # this is used to make sure that the initial value is 1.0 after softplus
      initializer = keras.initializers.Constant(np.float32(np.log(np.exp(1.0)-1)))
      self._scale = self.add_weight(name='negative_softmax_param',
                                    shape=(),
                                    dtype=tf.float32,
                                    initializer=initializer)
    super(UnaryModule, self).build(input_shapes)

  def call(self, inputs, training=None):
    '''
    Args:
      inputs: a list of two tensors
    '''
    assert(isinstance(inputs, list) and len(inputs) == 2),'inputs muse be a list of two tensors'
    # fea0 is [MBS*K, M, D]
    # neg_fea is [MBS, L, D]
    fea0, neg_fea = inputs
    if neg_fea is None:
      # [MBS*K, M, 1, 1]
      scores = tf.zeros(fea0.shape[:-1].as_list() + [1,1], dtype=tf.float32)
      return scores
    fea0_shape = fea0.shape.as_list()
    neg_shape = neg_fea.shape.as_list()
    pos_shape = fea0.shape.as_list()
    # pos_fea: [MBS*K, M, D] ==> [MBS, K*M, D]
    pos_fea = tf.reshape(fea0, [neg_shape[0], -1, pos_shape[-1]])
    pos_shape = pos_fea.shape.as_list()

    kwargs = {}
    # Only compute sim to topk nn in the negative bags
    if self._topk and self._fast_similarity is not None:
      fast_sim = self._fast_similarity([pos_fea, neg_fea], training=training)
      fast_sim = tf.stop_gradient(fast_sim[...,0])
      _, inds = tf.nn.top_k(fast_sim, self._topk, sorted=False)
      inds = tf.reshape(inds, [neg_shape[0], -1])
      neg_fea = util.batched_gather(inds, neg_fea)
      neg_fea = tf.reshape(neg_fea, pos_shape[:2] + [
                                    self._topk,
                                    neg_shape[-1]])
      kwargs['tile_second_fea'] = False
    scores = self._relation_unit([pos_fea, neg_fea], training=training, **kwargs)
    if self._mode == 'MAX':
      scores = tf.reduce_max(scores, axis=-2)
    elif self._mode == 'MEAN':
      scores = tf.reduce_mean(scores, axis=-2)
    elif self._mode == 'SOFTMAX':
      scale = tf.nn.softplus(self._scale)
      w = tf.nn.softmax(scale*scores[...,0])[..., tf.newaxis]
      scores = tf.reduce_sum(w*scores, axis=-2)

    # -scores since negative instances get higher score
    scores = tf.reshape(scores,fea0_shape[:-1])
    return scores

