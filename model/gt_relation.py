import tensorflow as tf
from tensorflow import keras

class GTRelation(object):
  def __call__(self, inputs, training=None):
    assert(isinstance(inputs, list) and len(inputs) == 2),'inputs must be a list of two tensors'
    #fea0 is N,M,D and fea1 is N,L,D
    cls0, cls1 = inputs

    cls0 = cls0[:, :, tf.newaxis]
    cls1 = cls1[:, tf.newaxis]

    r = tf.equal(cls0, cls1)
    labels = tf.cast(r, tf.float32)

    return labels


