# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import os

import tensorflow as tf

import transformer_model
import text_processor

INPUT_TENSOR_NAME = "inputs"
SIGNATURE_NAME = "serving_default"

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10
NUM_DATA_BATCHES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000 * NUM_DATA_BATCHES
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
RESNET_SIZE = 32
BATCH_SIZE = 1

# Scale the learning rate linearly with the batch size. When the batch size is
# 128, the learning rate should be 0.05.
_INITIAL_LEARNING_RATE = 0.05 * BATCH_SIZE / 128
_MOMENTUM = 0.9

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4

_BATCHES_PER_EPOCH = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE


def model_fn(features, labels, mode):

    prompts = features["prompt"]
    responses = features["responses"]
    vocab_size = features["vocab_size"]

    network = transformer_model.TED_generator(vocab_size)

    logits = network(prompts, responses, mode == tf.estimator.ModeKeys.TRAIN)

    predictions = {
        'classes': tf.argmax(logits, axis=2)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))  # Every element that is NOT padded
        # They will have to deal with run on sentences with this kind of setup
        loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    # Add weight decay to the loss.
    loss = loss_function(labels, logits)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(loss, name='loss')
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        optimizer = tf.train.AdamOptimizer(beta2=0.98, epsilon=1e-9)

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)


def make_batch(data_dir, batch_size=2):
    dataset = tf.data.TFRecordDataset(data_dir).repeat()

    dataset = dataset.map(parser, num_parallel_calls=batch_size)

    min_queue_examples = int(45000 * 0.4)
    # Ensure that the capacity is sufficiently large to provide good random
    # shuffling.
    dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()

    image, label = iterator.get_next()

    return image, label


def parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    image = tf.cast(tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]), tf.float32)
    label = tf.cast(features['label'], tf.int32)

    return image, label


def train_input_fn(data_dir):
    with tf.device('/cpu:0'):
        train_data = os.path.join(data_dir, 'train.tfrecords')
        image_batch, label_batch = make_batch(train_data, BATCH_SIZE)
        return {INPUT_TENSOR_NAME: image_batch}, label_batch


def eval_input_fn(data_dir):
    with tf.device('/cpu:0'):
        eval_data = os.path.join(data_dir, 'eval.tfrecords')
        image_batch, label_batch = make_batch(eval_data, BATCH_SIZE)

        return {INPUT_TENSOR_NAME: image_batch}, label_batch


def train(model_dir, data_dir, train_steps):

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    prompts, responses, tokenizer = text_processor(data_dir)
    vocab_size = int(tokenizer.vocab_size + 2)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"prompts": prompts, "responses": responses[:, :-1], "vocab_size": vocab_size},
        y=responses[:, 1:],
        batch_size=64,
        num_epochs=None,
        shuffle=True)

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)

    #temp_input_fn = functools.partial(train_input_fn, data_dir)

    #train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=train_steps)

    #exporter = tf.estimator.LatestExporter('Servo', serving_input_receiver_fn=serving_input_fn)
    #temp_eval_fn = functools.partial(eval_input_fn, data_dir)
    #eval_spec = tf.estimator.EvalSpec(temp_eval_fn, steps=1, exporters=exporter)

    #tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)
    estimator.train(
        input_fn=train_input_fn,
        steps=train_steps,
        hooks=[logging_hook])


def main(model_dir, data_dir, train_steps):
    tf.logging.set_verbosity(tf.logging.INFO)
    train(model_dir, data_dir, train_steps)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    # For more information:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    args_parser.add_argument(
        '--data-dir',
        default='/opt/ml/input/data/training',
        type=str,
        help='The directory where the CIFAR-10 input data is stored. Default: /opt/ml/input/data/training. This '
             'directory corresponds to the SageMaker channel named \'training\', which was specified when creating '
             'our training job on SageMaker')

    # For more information:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html
    args_parser.add_argument(
        '--model-dir',
        default='/opt/ml/model',
        type=str,
        help='The directory where the model will be stored. Default: /opt/ml/model. This directory should contain all '
             'final model artifacts as Amazon SageMaker copies all data within this directory as a single object in '
             'compressed tar format.')

    args_parser.add_argument(
        '--train-steps',
        type=int,
        default=100,
        help='The number of steps to use for training.')
    args = args_parser.parse_args()
    main(**vars(args))
