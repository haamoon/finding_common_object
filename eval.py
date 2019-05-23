import tensorflow as tf
import utils
from model.inference_model import InferenceModel
import os.path as osp
import logging
import argparse
from IPython import embed

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
  for step, dt in enumerate(dataset):
    inputs = utils.parse_dt(dt, config)
    top_subproblem, labels = model(inputs, training=False)
    utils.corloc(top_subproblem, labels, corloc_list)
    if (step+1) % config.print_freq == 0:
      logging.info('step {}/{}'.format(step+1, config.iters))
    if step >= config.iters:
      break

  mean_corloc = tf.reduce_mean(corloc_list).numpy()
  std_corloc = tf.math.reduce_std(corloc_list).numpy()
  print('- done.')
  logging.info('Accuracy is {} +- {}'.format(mean_corloc, std_corloc))
  return mean_corloc, std_corloc, corloc_list

def main():
  tf.config.gpu.set_per_process_memory_growth(True)
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

