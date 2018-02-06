# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Contains the TFExampleDecoder its associated helper classes.

The TFExampleDecode is a DataDecoder used to decode TensorFlow Example protos.
In order to do so each requested item must be paired with one or more Example
features that are parsed to produce the Tensor-based manifestation of the item.
"""

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.slim.python.slim.data.tfexample_decoder import ItemHandler
from tensorflow.python.framework import sparse_tensor


class MSCOCOKeypointsDecoder(ItemHandler):
  """An ItemHandler that concatenates a set of parsed Tensors to Bounding Boxes.
  """

  def __init__(self, keys=None, prefix='image/object/keypoints/'):
    """Initialize the MSCOCO Keypoint handler.

    Args:
      key: A name representing the mscoco keypoints
      (34 data points <= 17 keypoints represented with x, y coordinates)

    Raises:
      ValueError: if keys is not `None` and also not a list of exactly 4 keys
    """
    if keys is None:
      keys = [
        'ynose',        'xnose',
        'yleft_eye',    'xleft_eye',
        'yright_eye',   'xright_eye',
        'yleft_ear',    'xleft_ear',
        'yright_ear',   'xright_ear',
        'yleft_shoulder', 'xleft_shoulder',
        'yright_shoulder', 'xright_shoulder',
        'yleft_elbow',  'xleft_elbow',
        'yright_elbow', 'xright_elbow',
        'yleft_wrist',  'xleft_wrist',
        'yright_wrist', 'xright_wrist',
        'yleft_hip',    'xleft_hip',
        'yright_hip',   'xright_hip',
        'yleft_knee',   'xleft_knee',
        'yright_knee',  'xright_knee',
        'yleft_ankle',  'xleft_ankle',
        'yright_ankle', 'xright_ankle'
      ]
    elif len(keys) != 34:
      raise ValueError('BoundingBox expects 4 keys but got {}'.format(
          len(keys)))
    self._prefix = prefix
    self._keys = keys
    self._full_keys = [prefix + k for k in keys]
    super(MSCOCOKeypointsDecoder, self).__init__(self._full_keys)

  def tensors_to_item(self, keys_to_tensors):
    """Maps the given dictionary of tensors to a contatenated list of bboxes.

    Args:
      keys_to_tensors: a mapping of TF-Example keys to parsed tensors.

    Returns:
      [num_boxes, 17, 2] tensor of keypoints,
        i.e. 1 keypoint, in order [y, x].
    """
    tf_keypoints = []
    for full_keys in self._full_keys:
      mscoco_keypoints = keys_to_tensors[full_keys]
      if isinstance(mscoco_keypoints, sparse_tensor.SparseTensor):
        mscoco_keypoints = mscoco_keypoints.values
      mscoco_keypoints = array_ops.expand_dims(mscoco_keypoints, 1)
      tf_keypoints.append(mscoco_keypoints)
    tf_keypoints = tf.concat(tf_keypoints, axis=1)
    tf_keypoints = tf.reshape(tf_keypoints, [-1, 17, 2])
    return tf_keypoints