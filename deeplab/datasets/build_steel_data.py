
import io
import math
import numpy as np
import os
import pandas as pd
import random
import sys
import tensorflow as tf
from PIL import Image

import build_data


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_folder', '../data/train_images', 'Folder containing training images')

tf.app.flags.DEFINE_string('label_file', '../data/train.csv', 'Folder containing annotations for trainng images')

tf.app.flags.DEFINE_float('split_ratio', 0.9, 'Split ratio')

tf.app.flags.DEFINE_integer('seed', 42, 'Seed for reproducibility')

tf.app.flags.DEFINE_string('output_dir', '../data/tfrecord', 'Path to save converted tfrecord of Tensorflow example')

_NUM_SHARDS = 10


def numpy_to_bytes(image_np, image_format):
    image = Image.fromarray(image_np)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image_format)
    return image_bytes.getvalue()


def rle_to_mask(rle_string, height, width):
    if isinstance(rle_string, float) and math.isnan(rle_string):
        return np.zeros((height, width), dtype=np.uint8)
    
    rle_int = list(map(int, rle_string.split(' ')))
    rle_pair = np.array(rle_int).reshape(-1, 2)
    
    img = np.zeros(height * width, dtype=np.uint8)
    for index, length in rle_pair:
        index -= 1
        img[index: index+length] = 1
    
    img = img.reshape(width, height)
    img = img.T
    return img


def masks_to_mask(mask_list):
    assert len(mask_list) == 4, "The list of masks should be of length 4."
    mask = np.zeros(mask_list[0].shape[:2], np.uint8)
    
    for i in range(4):
      mask += (i + 1) * mask_list[i]

    return mask


def _split_dataset(image_folder, split_ratio):
  image_names = tf.gfile.Glob(os.path.join(image_folder, '*.jpg'))
  # image_ids = [image_file.split('/')[-1].split('.')[0] for image_file to image_files]
  
  # Shuffle image IDs
  random.seed(FLAGS.seed)
  random.shuffle(image_names)
  
  train_image_names = image_names[:int(len(image_names) * split_ratio)]
  val_image_names = image_names[int(len(image_names) * split_ratio):]
  
  return train_image_names, val_image_names


def _convert_dataset(dataset_split, image_names, labels_df):
  """Converts the ADE20k dataset into into tfrecord format.

  Args:
    dataset_split: Dataset split (e.g., train, val).
    image_ids: ...
    labels_df: ...

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  num_images = len(image_names)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))
  image_class_id_to_rle_mask = dict(zip(labels_df.ImageId_ClassId, labels_df.EncodedPixels))

  image_reader = build_data.ImageReader('jpeg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(FLAGS.output_dir, 
        '%s-%05d-of-%05d.tfrecord' % (dataset_split, shard_id, _NUM_SHARDS))
    
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i + 1, num_images, shard_id))
        sys.stdout.flush()
        # Read the image.
        image_name = image_names[i]
        image_data = tf.gfile.FastGFile(image_name, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)

        # Read the semantic segmentation annotation.
        image_id = image_name.split('/')[-1].split('.')[0]
        rle_masks = [image_class_id_to_rle_mask['{}.jpg_{}'.format(image_id, i+1)] for i in range(4)] 
        masks = [rle_to_mask(rle_mask, height, width) for rle_mask in rle_masks]
        mask = masks_to_mask(masks)
        mask_data = numpy_to_bytes(mask, 'png')

        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(image_data, image_name, height, width, mask_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  tf.gfile.MakeDirs(FLAGS.output_dir)
  FLAGS.image_format = 'jpeg'

  train_image_names, val_image_names = _split_dataset(FLAGS.image_folder, FLAGS.split_ratio)
  labels_df = pd.read_csv(FLAGS.label_file)

  _convert_dataset('train', train_image_names, labels_df)
  _convert_dataset('val', val_image_names, labels_df)


if __name__ == '__main__':
  tf.app.run()
