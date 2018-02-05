# Copyright 2017 Alexander Ponamarev. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the MSCOCO Keypoint Dataset (images + annotations).
"""
import os

import tensorflow as tf
from tensorflow.contrib import slim
from datasets import dataset_utils
from datasets.mscoco_keypoint_decoder import MSCOCOKeypointsDecoder

FILE_PATTERN = 'mscoco-.tfrecord'

VOC_LABELS = {
  'none': (0, 'Background'),
  'person': (1, 'Person'),
}

SPLITS_TO_SIZES = {
  'train': 32800,
  'val': 3280
}

ITEMS_TO_DESCRIPTIONS = {
  'image': 'A color image of varying height and width.',
  'shape': 'Shape of the image',
  'object/bbox': 'A list of bounding boxes, one per each object. [ymin, xmin, ymax, xmax]',
  'object/label': 'A list of labels, one per each object.',
  'object/keypoints': 'A list of 17 keypoints [y,x], one per each object'
}

def get_split(split_name, dataset_dir, file_pattern=FILE_PATTERN, reader=None,
              split_to_sizes=SPLITS_TO_SIZES, items_to_descriptions=ITEMS_TO_DESCRIPTIONS, num_classes=1):
  """Gets a dataset tuple with instructions for reading MSCOCO dataset.

  Args:
    split_name: A train/val split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
      ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in split_to_sizes:
      raise ValueError('split name %s was not recognized.' % split_name)
  file_pattern = os.path.join(dataset_dir, file_pattern)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
      reader = tf.TFRecordReader
  # Features in Pascal VOC TFRecords.
  keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
    'image/height': tf.FixedLenFeature([1], tf.int64),
    'image/width': tf.FixedLenFeature([1], tf.int64),
    'image/channels': tf.FixedLenFeature([1], tf.int64),
    'image/shape': tf.FixedLenFeature([3], tf.int64),
    'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
    'image/object/keypoints': tf.VarLenFeature(dtype=tf.float32)
  }
  items_to_handlers = {
    'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
    'shape': slim.tfexample_decoder.Tensor('image/shape'),
    'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['xmin', 'ymin', 'xmax', 'ymax'], 'image/object/bbox/'),
    'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
    'object/keypoints': MSCOCOKeypointsDecoder('image/object/keypoints')
  }
  decoder = slim.tfexample_decoder.TFExampleDecoder(
    keys_to_features, items_to_handlers)

  return slim.dataset.Dataset(
    data_sources=file_pattern,
    reader=reader,
    decoder=decoder,
    num_samples=split_to_sizes[split_name],
    items_to_descriptions=items_to_descriptions,
    num_classes=num_classes)

def main(*params):

  split_name = 'train'
  dataset_dir = '/Volumes/Data/Datasets/MSCOCO/'

  dataset = get_split(split_name, dataset_dir)
  provider = slim.dataset_data_provider.DatasetDataProvider(
    dataset,
    num_readers=1,
    common_queue_capacity=8,
    common_queue_min=16,
    shuffle=False)

  [image, glabels, gbboxes, keypoints] = provider.get(
    ['image', 'object/label', 'object/bbox', 'object/keypoints'])

  sess = tf.InteractiveSession()
  outputs = sess.run([image, glabels, gbboxes, keypoints])

  print("Done")

tf.app.run()