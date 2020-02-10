# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Efficient ImageNet input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.compat.v1 as tf

from tf_preprocessing import preprocess_image


def image_serving_input_fn():
  """Serving input fn for raw images."""

  def _preprocess_image(image_bytes):
    """Preprocess a single raw image."""
    image = preprocess_image(
        image_bytes=image_bytes, is_training=False)
    return image

  image_bytes_list = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  images = tf.map_fn(
      _preprocess_image, image_bytes_list, back_prop=False, dtype=tf.float32)
  return tf.estimator.export.ServingInputReceiver(
      images, {'image_bytes': image_bytes_list})


class ImageNetInput(object):
  """Generates ImageNet input_fn for training or evaluation.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py

  Args:
    is_training: `bool` for whether the input is for training.
    data_dir: `str` for the directory of the training and validation data.
    use_bfloat16: If True, use bfloat16 precision; else use float32.
    batch_size: The global batch size to use.
  """

  def __init__(self, is_training, data_dir, batch_size, num_classes, use_bfloat16=False, min_crop_size=0.2):
    self.image_preprocessing_fn = preprocess_image
    self.is_training = is_training
    self.use_bfloat16 = use_bfloat16
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.min_crop_size = min_crop_size

  def dataset_parser(self, value):
    """Parse an ImageNet record from a serialized string Tensor."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, ''),
        'image/class/label':
            tf.FixedLenFeature([], tf.int64, -1),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

    image = self.image_preprocessing_fn(
        image_bytes=image_bytes,
        is_training=self.is_training,
        use_bfloat16=self.use_bfloat16,
        min_crop_size=self.min_crop_size)

    # # Subtract one so that labels are in [0, 1000), and cast to float32 for
    # # Keras model.
    # label = tf.cast(tf.cast(
    #     tf.reshape(parsed['image/class/label'], shape=[1]), dtype=tf.int32) - 1,
    #                 dtype=tf.float32)

    label = tf.one_hot(parsed['image/class/label'] - 1, depth=self.num_classes)  # todo: one hot for label smoothing
    # label = parsed['image/class/label']

    return image, label

  def input_fn(self, ctx=None):
    """Input function which provides a single batch for train or eval.

    Args:
      ctx: Input context.

    Returns:
      A `tf.data.Dataset` object.
    """
    # Shuffle the filenames to ensure better randomization.
    try:
        file_pattern = os.path.join(
            self.data_dir, 'train-*' if self.is_training else 'validation-*')
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.is_training)
    except:
        file_pattern = os.path.join(
            self.data_dir, 'train-*' if self.is_training else 'val-*')
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.is_training)

    if ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)

    if self.is_training:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024     # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.interleave(
        fetch_dataset, cycle_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if self.is_training:
      dataset = dataset.shuffle(1024)

    # Parse, pre-process, and batch the data in parallel
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            self.dataset_parser,
            batch_size=self.batch_size,
            num_parallel_batches=2,
            drop_remainder=self.is_training))

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if self.is_training:
      options = tf.data.Options()
      options.experimental_deterministic = False
      dataset = dataset.with_options(options)

    return dataset
