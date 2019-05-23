import tensorflow as tf
import numpy as np
import time
from model.inference_model import InferenceModel
import utils
import logging
import argparse
import os
from eval import eval

parser = argparse.ArgumentParser()
parser.add_argument('--experiments_dir', required=True, help="Directory containing the experiments")

def train(config, train_dataset):
  logging.info('Starting training for {} iterations.'.format(config.train.iters))
  lr_schedule = utils.get_lr_schedule(config.train.lr_schedule)
  optimizer = tf.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
  model = InferenceModel(config.train)
  ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=model, optimizer=optimizer)
  manager = tf.train.CheckpointManager(ckpt, config.checkpoint,
                                       max_to_keep=3)

  ckpt.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    logging.info('Restored from {}'.format(manager.latest_checkpoint))
  else:
    logging.info('Initializing from scratch')

  def save_fn(manager, ckpt):
    save_path = manager.save()
    logging.info('Saved checkpoint for step {}: {}'.format(int(ckpt.step), save_path))
  best_acc = utils.get_best_acc(config.experiments_dir)

  try:
    summary_writer = tf.summary.create_file_writer(config.train.summary_dir)

    for dt in train_dataset:
      with tf.GradientTape() as tape:
        inputs = utils.parse_dt(dt, config.train)

        (pairwise_preds, pairwise_labels,
         unary_preds, unary_labels) = model(inputs, training=True)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=pairwise_labels,
                                                       logits=pairwise_preds)
        pairwise_loss = tf.reduce_mean(loss)
        regularization_loss = 0.5*tf.reduce_sum(model.losses)
        loss = pairwise_loss + regularization_loss

        if config.train.negative_bag_size > 0:
          unary_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                                                      labels=unary_labels,
                                                      logits=unary_preds)
          unary_loss = tf.reduce_mean(unary_loss)
          loss += unary_loss

      variables = model.trainable_variables
      grads = tape.gradient(loss, variables)
      optimizer.apply_gradients(zip(grads, variables))
      ckpt.step.assign_add(1)

      if int(ckpt.step) % config.train.print_freq == 0:
        print(ckpt.step.numpy(), loss.numpy())
        with summary_writer.as_default():
          step = int(ckpt.step)
          tf.summary.scalar('pairwise loss', pairwise_loss, step=step)
          tf.summary.scalar('regularization loss', regularization_loss, step=step)
          tf.summary.scalar('total loss', loss, step=step)
          tf.summary.scalar('learning rate', lr_schedule(float(step)), step=step)
          if config.train.negative_bag_size > 0:
            tf.summary.scalar('unary loss', unary_loss, step=step)
      if int(ckpt.step) % config.train.save_freq == 0:
        save_fn(manager, ckpt)
      if int(ckpt.step) % config.train.eval_freq == 0:
        #TODO: We have to load it every time because of rng issues
        logging.info('Loading eval dataset.')
        eval_dataset = utils.get_dataset(config.eval, training=False)
        logging.info('- done.')
        acc, _, _ = eval(config.eval, eval_dataset)
        with summary_writer.as_default():
          tf.summary.scalar('eval accuracy', acc, step=int(ckpt.step))
        if acc > best_acc:
          logging.info("New best accuracy found.")
          best_ckpt = tf.train.Checkpoint(step=ckpt.step, net=model,
                                          optimizer=optimizer)
          best_saver = tf.train.CheckpointManager(best_ckpt,
                                                  config.best_checkpoint,
                                                  max_to_keep=1)
          save_fn(best_saver, best_ckpt)
          utils.save_best_acc(config.experiments_dir, acc,
                              int(ckpt.step))
          best_acc = acc
      if ckpt.step >= config.train.iters:
        break

    save_fn(manager, ckpt)
  finally:
    summary_writer.close()

if __name__ == '__main__':
  tf.config.gpu.set_per_process_memory_growth(True)
  # Load the config from json file
  args = parser.parse_args()
  json_path = os.path.join(args.experiments_dir, 'config.json')
  assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
  config = utils.get_config(json_path)
  config.checkpoint = os.path.join(args.experiments_dir, 'train')
  config.train.summary_dir = os.path.join(args.experiments_dir, 'summary')
  config.best_checkpoint = os.path.join(args.experiments_dir, 'best')
  config.experiments_dir = args.experiments_dir
  utils.set_logger(os.path.join(args.experiments_dir, 'train.log'))
  logging.info('Loading the dataset...')
  #TODO: train dataset should always be created before eval due to rng
  train_dataset = utils.get_dataset(config.train, training=True)
  logging.info('- done.')
  train(config, train_dataset)

