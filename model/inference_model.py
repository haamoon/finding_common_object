import tensorflow as tf
from tensorflow import keras

from model.gt_relation import GTRelation
from model.relation_module import RelationModule
from model.cosine_similarity import CosineSimilarity
from model.unary_module import UnaryModule
from model.pairwise_cross_similarity import PairwiseCrossSimilarity

import numpy as np

from model.util import batched_gather

class InferenceModel(keras.Model):
  def __init__(self, config):
    super(InferenceModel, self).__init__()
    self._config = config

    self._gt_relation = GTRelation()
    self._relation_module = RelationModule(config.regularization_scale)
    if config.negative_bag_size > 0:
      unary_relation_module = RelationModule(config.regularization_scale)
      self._unary_module = UnaryModule(unary_relation_module)
      self._gt_unary_module = UnaryModule(self._gt_relation, mode='MAX')

  def call(self, inputs, training):
    assert(training is not None)
    if training:
      return self.training_call(inputs)
    else:
      return self.testing_call(inputs)

  def testing_call(self, inputs):
    training = False
    pos_fea, neg_fea, pos_classes, _, target_class = inputs
    pos_shape = tf.shape(pos_fea)
    pos_fea = tf.reshape(pos_fea, [-1, pos_shape[2] , pos_shape[3]])
    k = pos_shape[1]
    pos_shape = tf.shape(pos_fea)

    unary_energy = self._unary_module([pos_fea, neg_fea], training=training)
    tree_height = np.log2(k)
    assert(tree_height % 1.0 == 0.0), 'K is not power of 2.'
    tree_height = int(tree_height)

    subproblems = tf.range(pos_shape[1])[tf.newaxis, :, tf.newaxis]
    subproblems = tf.tile(subproblems, [pos_shape[0], 1, 1])

    pairwise_energy = tf.zeros_like(unary_energy)

    def split(tensor):
      return tensor[::2], tensor[1::2]

    def gather(pairs, tensor0, tensor1):
      return (batched_gather(pairs[..., 0], tensor0),
              batched_gather(pairs[..., 1], tensor1))

    unary_scales = self._config.unary_scales
    #[0.2, 0.5, 0.8, 1.6]
    ntop_proposal = self._config.ntop_proposals
    for i in range(tree_height):
      k = 2**(i+1)
      subproblems0, subproblems1 = split(subproblems)
      pairwise_energy0, pairwise_energy1 = split(pairwise_energy)
      unary_energy0, unary_energy1 = split(unary_energy)

      pairwise_module = PairwiseCrossSimilarity(self._relation_module, k)
      scores, pairs = pairwise_module([pos_fea, subproblems0, subproblems1])

      subproblems0, subproblems1 = gather(pairs, subproblems0, subproblems1)
      pairwise_energy0, pairwise_energy1 = gather(pairs, pairwise_energy0, pairwise_energy1)
      unary_energy0, unary_energy1 = gather(pairs, unary_energy0, unary_energy1)

      ## new pairwise energy = left side energy + right side energy + cross energy
      pairwise_energy = (pairwise_energy0 + pairwise_energy1 - scores[..., 0])

      unary_energy = unary_energy0 + unary_energy1

      # Subproblems energies
      total_energy = pairwise_energy + unary_scales[i] * unary_energy

      # All the subproblems 
      subproblems = tf.concat([subproblems0, subproblems1], axis=-1)

      # Sample topk proposals with the lowest energy
      topk = tf.minimum(tf.shape(total_energy)[-1], ntop_proposal)
      if i == tree_height - 1:
        topk = 1
      _, top_inds = tf.nn.top_k(-total_energy, sorted=True, k=topk)
      (subproblems, unary_energy,
       pairwise_energy) = batched_gather(top_inds, subproblems, unary_energy,
                                                   pairwise_energy)

    target_class = tf.cast(target_class, dtype=tf.float32)
    is_target = tf.cast(tf.equal(pos_classes, target_class[:, tf.newaxis, tf.newaxis]),
                        dtype=tf.float32)
    return subproblems[:, 0] , is_target

  def training_call(self, inputs):
    training = True
    pos_fea, neg_fea, pos_classes, neg_classes, _ = inputs
    pairwise_labels = self._gt_relation([pos_classes[:, 0, :, tf.newaxis],
                                         pos_classes[:, 1, :, tf.newaxis]])
    pairwise_preds = self._relation_module([pos_fea[:,0], pos_fea[:,1]],
                                            training=training)
    unary_preds = None
    unary_labels = None
    if self._config.negative_bag_size > 0:
      pcls = tf.reshape(pos_classes, [-1, self._config.bag_size, 1])
      ncls = neg_classes[..., tf.newaxis]
      unary_labels = self._gt_unary_module([pcls, ncls])

      shape = tf.shape(pos_fea)
      pfea = tf.reshape(pos_fea, [-1, shape[2], shape[3]])
      unary_preds = self._unary_module([pfea, neg_fea], training=training)

    return pairwise_preds, pairwise_labels, unary_preds, unary_labels
