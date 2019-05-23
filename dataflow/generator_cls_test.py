from generator_cls import GeneratorCLS
import tensorflow as tf

def test_with_tf_dataset():
  gen = GeneratorCLS(is_training=True, k_shot=4, bag_size=5, shuffle=True,
                     dataset_name='miniimagenet', split='train', use_features=True,
                     num_negative_bags=2, num_sample_classes=10, add_gt_list=False)
  gen.reset_state()
  dataset = tf.data.Dataset.from_generator(gen.get_data,
                                            (tf.float32, tf.float32, tf.float32,
                                             tf.float32, tf.float32, tf.int32))
  dataset = dataset.prefetch(50).batch(4).repeat()
  for i, d in enumerate(dataset):
    if i == 100:
      break
  from IPython import embed;embed()

if __name__ == '__main__':
  test_with_tf_dataset()
