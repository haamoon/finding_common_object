# coding: utf-8
import pickle
import tensorflow as tf

def load_python2_pickle(filename):
  with open(filename,'rb') as f:
    p = pickle.load(f, encoding='latin1')
  return p

def batched_gather(indices, *tensor_list):
  ''' Batched gather operation. For every tensor in the tensor_list
      output[i,j, ...] = tensor[i, indices[i, j], ...]
      Args:
        indices: array of indices to select with size [N, M]
        tensor_list: list of tensors with size [N, L, ...]
  '''

  assert(len(indices.shape) == 2)

  # create a 3D tensor with size [N, M, 2]
  # in which array[i, j] = (i, indices[i,j])
  n_ids = tf.range(indices.shape[0], dtype=indices.dtype)
  n_ids = tf.tile(n_ids[:, tf.newaxis], [1, indices.shape[1]])
  indices = tf.stack([n_ids, indices], axis=2)
  ret = []
  for tensor in tensor_list:
    out = None if tensor is None else tf.gather_nd(tensor, indices)
    ret.append(out)
  if len(ret) == 1:
    return ret[0]
  return ret
