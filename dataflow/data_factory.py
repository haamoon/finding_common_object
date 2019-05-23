from dataflow.datasets.miniimagenet_v2 import load_miniimagenet_data

class DataFactory(object):
  def get_data(is_training, dataset_name, split, use_features):
    original_imgs = None
    if dataset_name == 'miniimagenet':
        data = load_miniimagenet_data(split=split, is_training=is_training, use_features=use_features)
    else:
      raise ValueError('dataset_name {} is not implemented'.format(dataset_name))
    if is_training:
      x = data[0]
      y = data[2]
      original_imgs = data[-2] #original_img_train
    else:
      x = data[1]
      y = data[3]
      original_imgs = data[-1] #original_img_test
    if dataset_name == 'miniimagenet':
      if split == 'train':
        nr_classes = 64
      elif split == 'test':
        nr_classes = 20
      elif split == 'val':
        nr_classes = 16
      else:
        raise ValueError('split {} is not recognized for miniimagenet'.format(split))

    return x, y, nr_classes, original_imgs
