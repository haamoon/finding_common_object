class RandomIterator:
  """
  iterates over [0..N) and returns a number of random indices
  Args:
    N: number of indices to iterate on
  """
  def __init__(self, N, rng):
    assert N>=0, 'N should be greater than zero'
    self.indices = list(range(N))
    self.current_idx = 0
    self.N = N
    self._rng = rng
    self._shuffle()

  def _shuffle(self):
    """
    Does the shuffling operation.
    """
    self._rng.shuffle(self.indices)

  def getNextIndices(self, num=1):
    """
    returns `num` random indices
    Note: assumes num <= N
    """
    assert num > 0, 'requested number (%d) should be greater than zero' % num
    assert num <= self.N, 'too many items requested N is %d, num requested is %d' % (self.N, num)
    ret_indices = []
    if self.current_idx + num <= self.N:
      ret_indices += self.indices[self.current_idx:self.current_idx+num]
      self.current_idx += num
    else:
      ret_indices += self.indices[self.current_idx:]
      self._shuffle()
      num_remained = self.current_idx + num - self.N
      ret_indices += self.indices[:num_remained]
      self.current_idx = num_remained
    return ret_indices

