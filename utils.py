import tensorflow as tf
from dataflow.generator_cls import GeneratorCLS
import json
import logging
from easydict import EasyDict
from dataflow.utils import fix_rng_seed
from model.util import batched_gather
import os
import pickle

def get_lr_schedule(lr_schedule):
  boundaries = lr_schedule.boundaries
  values = lr_schedule.values
  return tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

def corloc(top_subproblem, labels, corloc_list):
  ntotal = tf.cast(tf.shape(top_subproblem)[-1], tf.float32)
  for i in range(tf.shape(labels)[0]):
    res = batched_gather(top_subproblem[i, ..., tf.newaxis],
                         labels[i])
    corloc_list.append(tf.reduce_sum(res)/ntotal)

def get_best_acc(dir_path):
  top_pkl_file = os.path.join(dir_path, 'top.pkl')
  if os.path.isfile(top_pkl_file):
    with open(top_pkl_file, 'rb') as f:
      top = pickle.load(f)
    return top['best']
  return 0.0

def save_best_acc(dir_path, best_acc, iteration):
  top_pkl_file = os.path.join(dir_path, 'top.pkl')
  top = {'best': best_acc, 'iteration': iteration}
  with open(top_pkl_file, 'wb') as f:
    pickle.dump(top, f)

def get_dataset(config, training):
  if config.shuffle is False:
    fix_rng_seed(config.seed)
  num_negative_bags = config.negative_bag_size // config.bag_size
  gen = GeneratorCLS(is_training=config.is_training, shuffle=config.shuffle,
                     add_gt_list=False, k_shot=config.k_shot, #TODO now cannot return the list now
                     bag_size=config.bag_size, num_negative_bags=num_negative_bags,
                     split=config.split, num_sample_classes=config.num_sample_classes,
                     num_sample_classes_min=config.num_sample_classes_min, use_features=config.use_features,
                     dataset_name=config.dataset_name, one_example_per_class=config.one_example_per_class,
                     has_single_target=config.has_single_target)
  gen.reset_state()
  dataset = tf.data.Dataset.from_generator(gen.get_data,
                                            (tf.float32, tf.float32, tf.float32,
                                             tf.float32, tf.float32, tf.int32)).prefetch(
                                                 config.prefetch_buffer_size)

  if training:
    return dataset.batch(config.meta_batch_size).repeat()
  else:
    return dataset.batch(1) #NOTE: we can repeat since new problems will be different

def parse_dt(dt, config):
    fea, _, _, classes, _, target_class = dt
    pos_fea = fea[:, :config.k_shot, :, 0, 0]
    neg_fea = fea[:, config.k_shot:, :, 0, 0]
    neg_shape = tf.shape(neg_fea)
    ## [MBS, N_NEG_BAGS, BAG_SIZE, D] ==> [MBS, N_NEG_BAGS*BAG_SIZE, D]
    neg_fea = tf.reshape(neg_fea, [neg_shape[0], -1, neg_shape[-1]])

    pos_classes = classes[:, :config.k_shot]
    neg_classes = classes[:, config.k_shot:]
    ## [MBS, N_NEG_BAGS, BAG_SIZE] ==> [MBS, N_NEG_BAGS*BAG_SIZE]
    neg_classes = tf.reshape(neg_classes, [neg_shape[0], -1])
    return pos_fea, neg_fea, pos_classes, neg_classes, target_class

def get_config(path):
  with open(path,'r') as f:
    return EasyDict(json.load(f))

def set_logger(log_path):
  """Set the logger to log info in terminal and file `log_path`.

  In general, it is useful to have a logger so that every output to the terminal is saved
  in a permanent file. Here we save it to `model_dir/train.log`.

  Example:
  ```
  logging.info("Starting training...")
  ```

  Args:
      log_path: (string) where to log
  """
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
