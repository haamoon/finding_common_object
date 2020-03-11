import tensorflow as tf
from tensorflow import keras

class CosineSimilarity(keras.layers.Layer):
  '''Cosine similarity class. It takes two 3D tensors with the same dimension and
      produces the relation scores between all pairwise elements of the input tensors.
  '''
  def __init__(self, *args, **kwargs):
    super(CosineSimilarity, self).__init__(*args, **kwargs)

  def build(self, input_shapes):
    '''
    input_shapes: list of two tensors shapes of N,M,D and N,L,D
    '''
    assert (isinstance(input_shapes, list) and len(input_shapes)==2),'input_shapes must be a list of two tensors'
    assert (input_shapes[0][-1] == input_shapes[1][-1]),'two tensors should have the same feature dimensions'

    self.D = input_shapes[0][-1]
    super(CosineSimilarity, self).build(input_shapes)

  @tf.function
  def call(self, inputs, training=None):
    assert(isinstance(inputs, list) and len(inputs) == 2),'inputs must be a list of two tensors'
    #fea0 is N,M,D and fea1 is N,L,D
    fea0, fea1 = inputs
    fea0 = tf.nn.l2_normalize(fea0, axis=-1)
    fea1 = tf.nn.l2_normalize(fea1, axis=-1)
    scores = tf.matmul(fea0, fea1, transpose_b=True)
    return scores[..., tf.newaxis]


