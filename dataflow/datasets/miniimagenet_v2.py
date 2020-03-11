import functools
import logging
import os

import numpy as np

import os
import sys
import pickle

# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

IMAGE_SIZE = 84
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
IMAGE_SHAPE_CHANNELS_FIRST = (3, IMAGE_SIZE, IMAGE_SIZE)

MINIIMAGENET_DATA_PATH = 'data/miniimagenet_v2'

def _load_data(data_path, split, is_training, use_features=False):
    """
    Notes:
      train_num: first train_num part of each class is considered as training data
        while the rest in the respective split will be the test data. This was
        being used in the old version where we used 500 images from the train split
        as training and and the remaining 100 images of every class as test images.
        Now, this number should be 600 when we are training and 0 otherwise.
    """
    train_num = 600 if is_training else 0

    if use_features:
      pkl_file = os.path.join(data_path, split+'_fea_v2.pkl')
    else:
      pkl_file = os.path.join(data_path, split+'_v2.pkl')

    img_pkl_file = None
    original_images = None
    if use_features:
      img_pkl_file = os.path.join(data_path, split+'_v2.pkl')
      if os.path.exists(img_pkl_file):
        logger.info("Loading original images of MiniImageNet")
        with open(img_pkl_file, 'rb') as f:
          img_data_dict = pickle.load(f, encoding='latin1')
        original_images = img_data_dict['images']
    assert os.path.exists(pkl_file), "pickle file {} does not exist".format(pkl_file)

    logger.info("loading MiniImageNet data")
    with open(pkl_file, 'rb') as f:
      data_dict = pickle.load(f, encoding='latin1')
    images = data_dict['images']
    indices = data_dict['indices']
    synsets = data_dict['synsets']

    test_num = 600 - train_num
    x_train,x_test,y_train,y_test = None,None,[],[]
    original_img_train, original_img_test = None, None
    train_indices = []
    test_indices = []

    for c, class_indices in enumerate(indices):
      if train_num > 0:
        train_indices += class_indices[:train_num]
        y_train += [c for _ in range(train_num)]
      if test_num > 0:
        test_indices  += class_indices[train_num:]
        y_test += [c for _ in range(test_num)]
    x_train = images[train_indices]
    y_train = np.asarray(y_train)
    x_test = images[test_indices]
    y_test = np.asarray(y_test)
    if original_images is not None:
      original_img_train = original_images[train_indices]
      original_img_test = original_images[test_indices]

    x_train_mean, x_train_std = None, None
    if not use_features:
      # This is the standard ResNet mean/std image normalization.
      #x_train_mean = np.mean(
      #    x_train, axis=(0, 1, 2), keepdims=True, dtype=np.float64,
      #).astype(np.float32)
      #x_train_std = np.std(
      #    x_train, axis=(0, 1, 2), keepdims=True, dtype=np.float64,
      #).astype(np.float32)

      x_train_mean = np.array([[[[120.09486995, 114.71146099, 102.83302023]]]], dtype=np.float32)
      x_train_std =  np.array([[[[72.51993986, 70.22868904, 74.08895542]]]], dtype=np.float32)
      if train_num > 0:
        x_train = (x_train - x_train_mean) / x_train_std
      if test_num > 0:
        x_test = (x_test - x_train_mean) / x_train_std

    logger.info("loaded Mini-ImageNet data")
    if train_num > 0:
      logger.info("training set size: {}".format(len(x_train)))
    if test_num > 0:
      logger.info("test set size: {}".format(len(x_test)))

    return (x_train, x_test, y_train, y_test,
           x_train_mean, x_train_std,
           train_indices, test_indices, indices, synsets,
           original_img_train, original_img_test)

# -----------------------------------------------------------------------------
load_miniimagenet_data = functools.partial(
    _load_data,
    data_path=MINIIMAGENET_DATA_PATH,
)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  pass
