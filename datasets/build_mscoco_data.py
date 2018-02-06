# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Converts MSCOCO data to TFRecords file format with Example protos.

This TensorFlow script converts the training and evaluation data into
a sharded data set.

  train_directory/mscoco-00000-of-TBD
  train_directory/mscoco-00001-of-TBD
  ...
  train_directory/mscoco-00127-of-TBD

and

  validation_directory/mscoco-00000-of-TBD
  validation_directory/mscoco-00001-of-TBD
  ...
  validation_directory/mscoco-00127-of-TBD

Each validation TFRecord file contains ~390 records. Each training TFREcord
file contains ~1250 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always'JPEG'

  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [1, 1000] where 0 is not used.
  image/class/text: string specifying the human-readable version of the label
    e.g. 'red fox, Vulpes vulpes'

  image/object/bbox/xmin: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/xmax: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/ymin: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/ymax: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/label: integer specifying the index in a classification
    layer. The label ranges from [1, 1000] where 0 is not used. Note this is
    always identical to the image label.
  image/object/keypoints/[34 points]

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO

tf.app.flags.DEFINE_string('data_directory', '/tmp/',
                           'Image data directory')
tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('shards', 64,
                            'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 4,
                            'Number of threads to preprocess the images.')


tf.app.flags.DEFINE_string('labels_file',
                           'annotations/person_keypoints_train2017.json',
                           'Labels file')

FLAGS = tf.app.flags.FLAGS


def _check_path(_p):
  if not tf.gfile.Exists(_p):
    raise FileNotFoundError("File doesn't exist on {} path.".format(_p))


def _check_type(v, t):
  if not isinstance(v, t):
    raise ValueError("Input of incorrect type. {} received while {} required".format(type(t).__name__, t.__name__))


def _create_data_catalog(path: str) -> tuple:
  """

  :param path: path to the person_keypoints dataset
  :return: MSCOCO imgs and annotations
  """
  _check_path(path)

  coco = COCO(path)
  cat = coco.loadCats(coco.getCatIds())[0]
  # get all images containing given categories, select one at random
  catIds = coco.getCatIds(catIds=cat['id'])
  imgIds = coco.getImgIds(catIds=catIds)
  imgs = coco.loadImgs(imgIds)
  img_anns = [coco.loadAnns(coco.getAnnIds(imgIds=_id, catIds=catIds, iscrowd=False))
              for _id in imgIds]

  return imgs, img_anns


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer, label, bbox,
                        keypoints, height, width):
  """Build an Example proto for an example.

  Args:
    image_buffer: string, JPEG encoding of RGB image
    label: integer list, identifier for the ground truth for the network
    bbox: list of bounding boxes; each box is a list of float
      specifying [xmin, ymin, xmax, ymax].
    keypoints: list of key points; each key point element contains 17 points specifying [x, y].
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto

  Note that x,y points for bounding boxes and keypoints are normalized by the image scale:
  x = x / image_width, y = y / image_height
  """
  xmin = []
  ymin = []
  xmax = []
  ymax = []

  xnose = []
  ynose = []
  xleft_eye = []
  yleft_eye = []
  xright_eye = []
  yright_eye = []
  xleft_ear = []
  yleft_ear = []
  xright_ear = []
  yright_ear = []
  xleft_shoulder = []
  yleft_shoulder = []
  xright_shoulder = []
  yright_shoulder = []
  xleft_elbow = []
  yleft_elbow = []
  xright_elbow = []
  yright_elbow = []
  xleft_wrist = []
  yleft_wrist = []
  xright_wrist = []
  yright_wrist = []
  xleft_hip = []
  yleft_hip = []
  xright_hip = []
  yright_hip = []
  xleft_knee = []
  yleft_knee = []
  xright_knee = []
  yright_knee = []
  xleft_ankle = []
  yleft_ankle = []
  xright_ankle = []
  yright_ankle = []
  for b, k in zip(bbox, keypoints):
    assert len(b) == 4
    assert len(k) == 34
    # pylint: disable=expression-not-assigned
    [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
    [l.append(point) for l, point in zip([
      xnose, ynose, xleft_eye, yleft_eye, xright_eye, yright_eye, xleft_ear,
      yleft_ear, xright_ear, yright_ear, xleft_shoulder, yleft_shoulder,
      xright_shoulder, yright_shoulder, xleft_elbow, yleft_elbow, xright_elbow,
      yright_elbow, xleft_wrist, yleft_wrist, xright_wrist, yright_wrist, xleft_hip,
      yleft_hip, xright_hip, yright_hip, xleft_knee, yleft_knee, xright_knee, yright_knee,
      xleft_ankle, yleft_ankle, xright_ankle, yright_ankle], k)]
    # pylint: enable=expression-not-assigned

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
    'image/encoded': _bytes_feature(image_buffer),
    'image/format': _bytes_feature(image_format.encode('utf-8')),
    'image/shape': _int64_feature([height, width, channels]),
    'image/channels': _int64_feature(channels),
    'image/height': _int64_feature(channels),
    'image/width': _int64_feature(channels),
    'image/colorspace': _bytes_feature(colorspace.encode('utf-8')),
    'image/object/bbox/xmin': _float_feature(xmin),
    'image/object/bbox/xmax': _float_feature(xmax),
    'image/object/bbox/ymin': _float_feature(ymin),
    'image/object/bbox/ymax': _float_feature(ymax),
    'image/object/bbox/label': _int64_feature(label),
    'image/object/keypoints/ynose': _float_feature(ynose),
    'image/object/keypoints/xnose': _float_feature(xnose),
    'image/object/keypoints/yleft_eye': _float_feature(yleft_eye),
    'image/object/keypoints/xleft_eye': _float_feature(xleft_eye),
    'image/object/keypoints/yright_eye': _float_feature(yright_eye),
    'image/object/keypoints/xright_eye': _float_feature(xright_eye),
    'image/object/keypoints/yleft_ear': _float_feature(yleft_ear),
    'image/object/keypoints/xleft_ear': _float_feature(xleft_ear),
    'image/object/keypoints/yright_ear': _float_feature(yright_ear),
    'image/object/keypoints/xright_ear': _float_feature(xright_ear),
    'image/object/keypoints/yleft_shoulder': _float_feature(yleft_shoulder),
    'image/object/keypoints/xleft_shoulder': _float_feature(xleft_shoulder),
    'image/object/keypoints/yright_shoulder': _float_feature(yright_shoulder),
    'image/object/keypoints/xright_shoulder': _float_feature(xright_shoulder),
    'image/object/keypoints/yleft_elbow': _float_feature(yleft_elbow),
    'image/object/keypoints/xleft_elbow': _float_feature(xleft_elbow),
    'image/object/keypoints/yright_elbow': _float_feature(yright_elbow),
    'image/object/keypoints/xright_elbow': _float_feature(xright_elbow),
    'image/object/keypoints/yleft_wrist': _float_feature(yleft_wrist),
    'image/object/keypoints/xleft_wrist': _float_feature(xleft_wrist),
    'image/object/keypoints/yright_wrist': _float_feature(yright_wrist),
    'image/object/keypoints/xright_wrist': _float_feature(xright_wrist),
    'image/object/keypoints/yleft_hip': _float_feature(yleft_hip),
    'image/object/keypoints/xleft_hip': _float_feature(xleft_hip),
    'image/object/keypoints/yright_hip': _float_feature(yright_hip),
    'image/object/keypoints/xright_hip': _float_feature(xright_hip),
    'image/object/keypoints/yleft_knee': _float_feature(yleft_knee),
    'image/object/keypoints/xleft_knee': _float_feature(xleft_knee),
    'image/object/keypoints/yright_knee': _float_feature(yright_knee),
    'image/object/keypoints/xright_knee': _float_feature(xright_knee),
    'image/object/keypoints/yleft_ankle': _float_feature(yleft_ankle),
    'image/object/keypoints/xleft_ankle': _float_feature(xleft_ankle),
    'image/object/keypoints/yright_ankle': _float_feature(yright_ankle),
    'image/object/keypoints/xright_ankle': _float_feature(xright_ankle),
    }))

  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  image_data = tf.gfile.FastGFile(filename, 'rb').read()

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width

def _process_bbox(bbox, height, width):
  """
  Converts MSCOCO keypoints [xmin, ymin, width, height] into
  Tensorflow format [xmin, ymin, xmax, ymax]
  x, y points are normalized with respect to image scale: x / image_width, y / image_height
  :param bbox: [xmin, ymin, width, height]
  :param height: image height
  :param width: image width
  :return: [xmin, ymin, xmax, ymax]
  """
  xmin, ymin, bw, bh = bbox
  xmax = xmin + bw
  ymax = ymin + bh
  ymin, ymax = ymin / height, ymax / height
  xmin, xmax = xmin / width, xmax / width

  return [xmin, ymin, xmax, ymax]

def _process_keypoints(keypoints, height, width):
  """
  :param keypoints: List of key points; each key point element
  contains 17 points specifying [x, y, visibility].
  :return: List of key points; each key point  specifying [x,y]
    x, y points are normalized with respect to image scale: x / image_width, y / image_height
    x, y for absent points are float('Nan'), float('Nan')
  """

  _normalize = lambda x,y,v: [x / height, y / width] if v > 0 else [float('Nan'), float('Nan')]

  x = keypoints[0::3]
  y = keypoints[1::3]
  v = keypoints[2::3]

  filtered_keypoints = []
  _ = list(
    map(lambda args: filtered_keypoints.extend(_normalize(*args)), zip(x, y, v))
  )

  return filtered_keypoints

def _process_image_files_batch(coder, thread_index, ranges, name, imgs, img_anns, num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    ims: MSCOCO image description containing file name, height, width
    img_anns: MSCOCO annotations grouped with respect to images (ims), containing
      bounding boxes [xmin, ymin, width, height] and keypoints (17 points for each person [x,y,visibility]).
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d.tfrecord' % ('mscoco', shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      img = imgs[i]
      anns = img_anns[i]
      filename = img['file_name']
      bbox = [ann['bbox'] for ann in anns]
      label = [ann['category_id'] for ann in anns]
      keypoints = [ann['keypoints'] for ann in anns]
      full_filename = os.path.join(FLAGS.data_directory, filename)

      try:

        image_buffer, height, width = _process_image(full_filename, coder)
        bbox = list(map(lambda b: _process_bbox(b, height, width), bbox))
        keypoints = list(map(lambda k: _process_keypoints(k, height, width), keypoints))

        example = _convert_to_example(image_buffer, label,
                                      bbox, keypoints, height, width)
        writer.write(example.SerializeToString())
        shard_counter += 1
        counter += 1

        if not counter % 1000:
          print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                (datetime.now(), thread_index, counter, num_files_in_thread))
          sys.stdout.flush()

      except:
        shard_counter += 1

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, imgs, img_anns, num_shards):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: list of integer; each integer identifies the ground truth
    humans: list of strings; each string is a human-readable label
    bboxes: list of bounding boxes for each image. Note that each entry in this
      list might contain from 0+ entries corresponding to the number of bounding
      box annotations for the image.
    num_shards: integer number of shards for this data set.
  """
  assert len(imgs) == len(img_anns)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(imgs), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, imgs, img_anns, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(imgs)))
  sys.stdout.flush()


def _find_image_files(data_dir, labels_file):
  """Build a list of all images files and labels in the data set.

  Args:
    data_dir: string, path to the root directory of images.

      Assumes that the ImageNet data set resides in JPEG files located in
      the following directory structure.

        data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
        data_dir/n01440764/ILSVRC2012_val_00000543.JPEG

      where 'n01440764' is the unique synset label associated with these images.

    labels_file: string, path to the labels file.

      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        n01440764
        n01443537
        n01484850
      where each line corresponds to a label expressed as a synset. We map
      each synset contained in the file to an integer (based on the alphabetical
      ordering) starting with the integer 1 corresponding to the synset
      contained in the first line.

      The reason we start the integer labels at 1 is to reserve label 0 as an
      unused background class.

  Returns:
    filenames: list of strings; each string is a path to an image file.
    synsets: list of strings; each string is a unique WordNet ID.
    labels: list of integer; each integer identifies the ground truth.
  """
  print('Determining list of input files and labels from %s.' % data_dir)
  challenge_synsets = [l.strip() for l in
                       tf.gfile.FastGFile(labels_file, 'r').readlines()]

  labels = []
  filenames = []
  synsets = []

  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of JPEG files and labels.
  for synset in challenge_synsets:
    jpeg_file_path = '%s/%s/*.JPEG' % (data_dir, synset)
    matching_files = tf.gfile.Glob(jpeg_file_path)

    labels.extend([label_index] * len(matching_files))
    synsets.extend([synset] * len(matching_files))
    filenames.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(challenge_synsets)))
    label_index += 1

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = range(len(filenames))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  synsets = [synsets[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(challenge_synsets), data_dir))
  return filenames, synsets, labels


def _process_dataset(label_path, directory, num_shards):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
  """
  imgs, img_anns = _create_data_catalog(label_path)
  _process_image_files(directory, imgs, img_anns, num_shards)


def main(unused_argv):
  print('Saving results to %s' % FLAGS.output_directory)

  # Run it!
  _process_dataset(FLAGS.labels_file, FLAGS.data_directory, FLAGS.shards)


if __name__ == '__main__':
  tf.app.run()
