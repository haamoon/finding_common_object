# coding: utf-8
import numpy as np
from data_sampler_cls import DataSamplerCLS
rng = np.random.RandomState(1357)

def test_sample_creation():
  sampler = DataSamplerCLS(rng, split='test', k_shot=8, bag_size=5, use_features=True, num_negative_bags=2, num_sample_classes_min=5, num_sample_classes=10)
  from IPython import embed;embed()

if __name__ == '__main__':
  test_sample_creation()
  print('passed')
