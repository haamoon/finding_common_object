import numpy as np
from dataflow.random_iterator import RandomIterator
from dataflow.data_factory import DataFactory

#Note this is the old DataSamplerWRN
class DataSamplerCLS(object):
  def __init__(self,
               rng,
               is_training=True,
               k_shot=4,
               bag_size=10,
               has_bag_image_iterator=True,
               dataset_name='miniimagenet',
               split='train',
               use_features=False,
               has_single_target=False,
               num_negative_bags=0,
               num_sample_classes=0,
               num_sample_classes_min=5,
               one_example_per_class=False):

    if num_sample_classes_min is None and num_sample_classes > 0:
      num_sample_classes_min = bag_size
    self.one_example_per_class = one_example_per_class
    self.num_sample_classes_min = num_sample_classes_min
    self.is_training = is_training
    self.k_shot = k_shot
    self.bag_size = bag_size
    self.has_bag_image_iterator = has_bag_image_iterator
    self.dataset_name = dataset_name
    self.has_single_target = has_single_target
    self.num_negative_bags = num_negative_bags
    self.num_sample_classes = num_sample_classes
    self.dataset_name = dataset_name
    self.original_imgs = None
    self.total_bags = k_shot + num_negative_bags

    x, y, nr_classes, original_imgs = DataFactory.get_data(is_training, dataset_name,
                                                           split, use_features)
    self.x = x
    self.y = y
    self.nr_classes = nr_classes
    self.original_imgs = original_imgs
    self.nr_images = x.shape[0]

    self.class_indices = [[] for _ in range(self.nr_classes)]
    self._fill_class_indices()
    self.class_iterators = [RandomIterator(len(x), rng) for x in self.class_indices]
    self.image_iterator = RandomIterator(self.nr_images, rng)
    self.bag_image_iterator = RandomIterator(self.nr_images, rng) #It might be repeated images
    if self.bag_size > 1:
      self.bag_iterator = RandomIterator(self.bag_size, rng)

    if self.num_sample_classes > 1:
      assert self.num_sample_classes >= self.num_sample_classes_min, 'num_sample_classes is not enough'
      self.sample_class_iterator = RandomIterator(self.nr_classes, rng)
      self.sample_classes_range = range(self.num_sample_classes_min, self.num_sample_classes+1)
      self.sample_classes_range_iterator = RandomIterator(len(self.sample_classes_range), rng)

  def _fill_class_indices(self):
    for i, class_idx in enumerate(self.y):
      self.class_indices[class_idx].append(i)

  @property
  def _next_class(self):
    indices = self.image_iterator.getNextIndices(num = 1)
    class_idx = self.y[indices[0]]
    return class_idx

  def _next_class_data(self, class_idx):
    current_class_index = self.class_indices[class_idx]
    idx = self.class_iterators[class_idx].getNextIndices(num = 1)
    current_idx = current_class_index[idx[0]]
    image = self.x[current_idx]
    label = self.y[current_idx]
    original_image = None
    if self.original_imgs is not None:
      original_image = self.original_imgs[current_idx]
    return image, label, original_image

  def sample_bag(self, img_iterator, size, class_idx, is_positive=True, sampled_classes=None):
    imgs = []
    labels = []
    original_images = None
    if self.original_imgs is not None:
      original_images = []
    while len(imgs) < size:
      index = img_iterator.getNextIndices(num=1)[0]
      label = self.y[index]
      if sampled_classes is not None and not label in sampled_classes:
        continue
      if label == class_idx:
        if not is_positive or self.has_single_target:
          continue
      if self.one_example_per_class and label in labels:
        continue
      labels.append(label)
      imgs.append(self.x[index])
      if self.original_imgs is not None:
        original_images.append(self.original_imgs[index])
    return imgs, labels, original_images

  def sample_other_classes(self, class_idx):
    sampled_classes = [] #list(set(self.sample_class_iterator.getNextIndices(self.num_sample_classes-1)))
    next_range_index = self.sample_classes_range_iterator.getNextIndices(1)[0]
    cur_num_sample_classes = self.sample_classes_range[next_range_index]

    while len(sampled_classes) != cur_num_sample_classes - 1:
      #THIS IS DUE TO AN ERORR (same class can come in sampled_class twice)
      other_class = self.sample_class_iterator.getNextIndices(1)[0]
      if other_class not in sampled_classes:
        sampled_classes.append(other_class)
    if class_idx in sampled_classes:
      #class_idx is there. 
      #so, we should sample another class which is not there
      other_class = self.sample_class_iterator.getNextIndices(1)[0]
      while other_class in sampled_classes:
        other_class = self.sample_class_iterator.getNextIndices(1)[0]
      sampled_classes.append(other_class)
    else:
      #class_idx is not there.
      #So, we should add it
      sampled_classes.append(class_idx)
    assert len(sampled_classes) == cur_num_sample_classes,'sampled_classes does not have correct len'
    assert class_idx in sampled_classes, 'class_idx should be in sampled_classes'
    return sampled_classes


  def next(self):
    """
    define:
      total_bags: k_shot + num_negative_bags
      positive_bags: k_shot
      num_images: total_bags*bag_size
    Returns:
      imgs: list of num_images numpy arrays.
        if use_features is True it has the shape (1,1, D) where D is
        the feature dimension
      labels: list of num_images integer labels where 1s correspond the common object
        and 0s correspond to non-common objects
      original_labels: list of num_images integers corresponding to each image label
        starting from zero
      original_imgs:
        num_images numpy arrays of shape (W,H,C) and type uint8 representing the
        original images.
      class_idx: id of the common class (starts from 0)
    """
    #print('next called')
    class_idx = self._next_class
    sampled_classes = None
    if self.num_sample_classes > 1:
      sampled_classes = self.sample_other_classes(class_idx)
    #print('Sampled classes: {}'.format(sampled_classes))
    imgs = []
    labels = []
    original_imgs = []
    for i in range(self.total_bags):
      img, label, original_img = self._next_class_data(class_idx)
      if self.bag_size > 1:
        bag_imgs = [img]
        bag_labels = [label]
        bag_original_imgs = [original_img]
        img_iterator = self.bag_image_iterator if self.has_bag_image_iterator else self.image_iterator
        if i < self.k_shot: #positive part
          (sampled_imgs, sampled_labels,
              sampled_original_imgs) = self.sample_bag( img_iterator,
                  self.bag_size - 1, class_idx,
                  is_positive=True, sampled_classes=sampled_classes)
          bag_imgs.extend(sampled_imgs)
          bag_labels.extend(sampled_labels)
          if self.original_imgs is not None:
            bag_original_imgs.extend(sampled_original_imgs)
        else: #negative part
          (bag_imgs, bag_labels,
              bag_original_imgs) = self.sample_bag( img_iterator, self.bag_size,
                class_idx, is_positive=False, sampled_classes=sampled_classes)

        indices = self.bag_iterator.getNextIndices(num=self.bag_size)
        imgs.extend([bag_imgs[j] for j in indices])
        labels.extend([bag_labels[j] for j in indices])
        if self.original_imgs is not None:
          original_imgs.extend([bag_original_imgs[j] for j in indices])
      else:
        if i < self.k_shot: #positive part
          imgs.append(img)
          labels.append(label)
          if self.original_imgs is not None:
            original_imgs.append(original_img)
        else: #negative part
          img_iterator = self.bag_image_iterator if self.has_bag_image_iterator else self.image_iterator
          (neg_img, neg_label,
              neg_original_img) = self.sample_bag(img_iterator, 1, class_idx,
                  is_positive=False, sampled_classes=sampled_classes)
          imgs.append(neg_img)
          labels.append(neg_label)
          if self.original_imgs is not None:
            original_imgs.append(neg_original_img)
    original_labels = [l for l in labels]
    labels = [int(l==class_idx) for l in labels]
    return imgs, labels, original_labels, original_imgs, class_idx

