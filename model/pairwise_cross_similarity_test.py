import tensorflow as tf
from pairwise_cross_similarity import PairwiseCrossSimilarity
from relation_module import RelationModule
from cosine_similarity import CosineSimilarity
import util
import numpy as np

def test_shape():
  rm = RelationModule()
  k = 2
  pw = PairwiseCrossSimilarity(rm, k)
  orig_fea = tf.random.uniform([8,5,10])
  ind = tf.tile(tf.range(orig_fea.shape[1])[tf.newaxis], [orig_fea.shape[0],1])
  ind = ind[..., tf.newaxis]
  ind0 = ind[0::2]
  ind1 = ind[1::2]
  scores, pairs = pw([orig_fea, ind0, ind1])

def test_input_outputs():
  io_dict = util.load_python2_pickle('pwio.pkl')
  cs = CosineSimilarity()
  for k in [2,4,8]:
    solo_fea = io_dict['{}_solo_fea'.format(k)]
    ind0 = io_dict['{}_ind0'.format(k)]
    ind1 = io_dict['{}_ind1'.format(k)]
    scores = io_dict['scores_{}'.format(k)]
    pw = PairwiseCrossSimilarity(cs, k)
    out_scores, out_pairs = pw([tf.constant(solo_fea),
                                tf.constant(ind0),
                                tf.constant(ind1)])
    assert(np.allclose(out_scores.numpy(), scores))

if __name__ == '__main__':
  test_input_outputs()
  print('Passed')
