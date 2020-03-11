import tensorflow as tf
import utils
from model.inference_model import InferenceModel
import os.path as osp
import logging
import argparse
from IPython import embed
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--experiments_dir', required=True,
          help="Directory containing the experiments")

def eval(config, dataset):
  logging.info('Running evaluation for {} iterations'.format(config.iters))
  model = InferenceModel(config)
  checkpoint = config.checkpoint
  if osp.isdir(config.checkpoint):
    checkpoint = tf.train.latest_checkpoint(config.checkpoint)

  logging.info('Restoring parameters from {}'.format(checkpoint))
  ckpt = tf.train.Checkpoint(net=model)
  ckpt.restore(checkpoint)
  corloc_list = []
  inference_time_list = []
  for step, dt in enumerate(dataset):
    inputs = utils.parse_dt(dt, config)
    start_time = time.time()
    top_subproblem, labels = model(inputs, training=False)
    ##
    #pos_fea, neg_fea, pos_classes, _, target_class = inputs
    #k, pos_fea = model.get_k_pos_fea(pos_fea)
    #unary_energy = model.get_unaries(pos_fea, neg_fea)
    #subproblems = model.get_top_selection(pos_fea, unary_energy, k)
    #is_target = model.get_is_target(target_class, pos_classes)
    #top_subproblem = subproblems[:,0]
    #labels = is_target
    ##
    inference_time_list.append(time.time() - start_time)
    utils.corloc(top_subproblem, labels, corloc_list)
    if (step+1) % config.print_freq == 0:
      logging.info('step {}/{} ({:0.3f} sec/iter)'.format(step+1, config.iters,
                            np.mean(inference_time_list[-config.print_freq:])))
    if step >= config.iters:
      break

  mean_corloc = tf.reduce_mean(corloc_list).numpy()
  std_corloc = tf.math.reduce_std(corloc_list).numpy()
  print('- done.')
  logging.info('Accuracy is {} +- {}'.format(mean_corloc, std_corloc))
  return mean_corloc, std_corloc, corloc_list

def main():
  #tf.config.gpu.set_per_process_memory_growth(True)
  # Load the config from json file
  args = parser.parse_args()
  json_path = osp.join(args.experiments_dir, 'config.json')
  assert osp.isfile(json_path), "No json configuration file found at {}".format(json_path)
  config = utils.get_config(json_path)
  utils.set_logger(osp.join(args.experiments_dir, 'test.log'))
  logging.info('Loading the dataset...')
  test_dataset = utils.get_dataset(config.test, training=False)
  logging.info('- done.')
  eval(config.test, test_dataset)

if __name__ == '__main__':
  main()

