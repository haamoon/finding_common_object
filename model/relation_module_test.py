import tensorflow as tf
from relation_module import RelationModule

def test_output_shape():
  r = RelationModule()
  x = tf.random.uniform([3,5,10])
  y = tf.random.uniform([3,7,10])
  s = r([x,y])
  assert s.shape.as_list() == [3,5,7,1]


if __name__ == '__main__':
  test_output_shape()
  print('Passed')
