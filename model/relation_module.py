import tensorflow as tf
from tensorflow import keras

class RelationModule(keras.layers.Layer):
  '''Relation module class. It takes two 3D tensors with the same dimension and
      produces the relation scores between all pairwise elements of the input tensors.
  '''
  def __init__(self, regularization_scale=0.0005):
    '''
    Args:
      regularization_scale: regularization constant
    '''
    super(RelationModule, self).__init__()
    self._regularization_scale = regularization_scale

  def build(self, input_shapes):
    '''
    input_shapes: list of two tensors shapes of N,M,D and N,L,D
    '''
    assert (isinstance(input_shapes, list) and len(input_shapes)==2),'input_shapes must be a list of two tensors'
    assert (input_shapes[0][-1] == input_shapes[1][-1]),'two tensors should have the same feature dimensions'

    self.D = input_shapes[0][-1]
    self.tanh_fc = keras.layers.Dense(self.D, input_dim=self.D*2,
        activation=None,
        use_bias=False,
        kernel_initializer='glorot_uniform',
        kernel_regularizer=keras.regularizers.l2(self._regularization_scale))
    self.tanh_fc.build(input_shape=[None,None,None,self.D*2])
    self.sigm_fc = keras.layers.Dense(self.D, input_dim=self.D*2,
        activation=None,
        use_bias=False,
        kernel_initializer='glorot_uniform',
        kernel_regularizer=keras.regularizers.l2(self._regularization_scale))
    self.sigm_fc.build(input_shape=[None,None,None,self.D*2])
    self.score_fc = keras.layers.Dense(1, input_dim=self.D,
        activation=None,
        kernel_initializer='glorot_uniform',
        kernel_regularizer=keras.regularizers.l2(self._regularization_scale))
    self.score_fc.build(input_shape=[None,None,None,self.D])
    self.tanh_bn = keras.layers.BatchNormalization(epsilon=1e-5)
    self.tanh_bn.build(input_shape=[None,None,None,self.D])
    self.sigm_bn = keras.layers.BatchNormalization(epsilon=1e-5)
    self.sigm_bn.build(input_shape=[None,None,None,self.D])
    super(RelationModule, self).build(input_shapes)

  def call(self, inputs, training=None, tile_second_fea=True):
    assert(isinstance(inputs, list) and len(inputs) == 2),'inputs must be a list of two tensors'
    #fea0 is N,M,D and fea1 is N,L,D
    fea0, fea1 = inputs
    m = fea0.shape[1] #Note: is this for eager mode only?

    #N,M,L,D
    if tile_second_fea:
      fea1 = tf.tile(fea1[:,tf.newaxis],[1,m,1,1])
    l = fea1.shape[2]
    fea0 = tf.tile(fea0[:,:,tf.newaxis], [1,1,l,1])

    #N,M,L,2D
    fea01 = tf.concat((fea0, fea1), axis=-1)
    #N,M,L,D
    out_tanh = self.tanh_fc(fea01)
    #TODO in the paper's code have bn and then tanh
    out_tanh = self.tanh_bn(out_tanh, training)
    out_tanh = tf.tanh(out_tanh)
    out_sigm = self.sigm_fc(fea01)
    out_sigm = self.sigm_bn(out_sigm, training)
    out_sigm = tf.sigmoid(out_sigm)
    #N,M,L,D
    C = out_tanh*out_sigm + (fea0+fea1)/2
    #N,M,L,1
    scores = self.score_fc(C)
    return scores


