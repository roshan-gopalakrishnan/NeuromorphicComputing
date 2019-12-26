#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: HFNetv1

import argparse
import os

import tensorflow as tf
import numpy as np
from tensorpack import *
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from imagenet_utils import (
    ImageNetModel, get_imagenet_dataflow, fbresnet_augmentor)


# conv module
def convnormrelu2D(x, name, chan, stride):
    x = tf.pad(x, [[0, 0], [0,0], [1, 1], [1, 1]])
    x = Conv2D(name, x, filters = chan, kernel_size = 3, strides=stride, padding='VALID')  #see conv2d.py
    x = BatchNorm(name + '_bn', x)
    x = tf.nn.relu(x, name=name + '_relu')
    return x

# Depthwise conv module
def convnormreluDW(x, name, filters, stride):
    x = tf.pad(x, [[0, 0], [0,0], [1, 1], [1, 1]])
    x = ConvDW(name, x, depthwise_filter = filters, stride=stride, padding='VALID') #depthwise conv
    x = BatchNorm(name + '_bn', x)
    x = tf.nn.relu(x, name=name + '_relu')
    return x

#Pointwise conv module
def convnormreluPW(x, name, filters):
    x = ConvPW(name, x, pointwise_filter = filters) # pointwise conv
    x = BatchNorm(name + '_bn', x)
    x = tf.nn.relu(x, name=name + '_relu')
    return x

class Model(ImageNetModel):
    weight_decay = 5e-4

    def get_logits(self, image):
        with argscope(Conv2D, use_bias=False, kernel_initializer=tf.variance_scaling_initializer(scale=2.)), \
                argscope([ConvDW, ConvPW], use_bias=False, W_init=tf.variance_scaling_initializer(scale=2.), data_format='NCHW'), \
                argscope([Conv2D, BatchNorm, MaxPooling], data_format='channels_first'):
            logits = (LinearWrap(image)
                      .apply(convnormrelu2D, 'conv0', 16, (2,2))  # (function, name, output_channle, (stride,stride))
                      .apply(convnormrelu2D, 'conv1', 28, (2,2))                      
                      .apply(convnormrelu2D, 'conv2', 64, (2,2))
                      
                      .apply(convnormreluDW, 'conv3_1', [3,3,64,1], 1) #(function, name, [kernel_size,kernel_size,output_channel,channel_multiplier],stride) 
                      .apply(convnormreluPW, 'conv3_2', [1,1,64,256])#(function, name, [kernel_size,kernel_size,input_channel,output_channel])

                      .apply(convnormreluDW, 'conv4_1', [3,3,256,1], 2)
                      .apply(convnormreluPW, 'conv4_2', [1,1,256,256])
                     
                      .apply(convnormreluDW, 'conv5_1', [3,3,256,1], 1)
                      .apply(convnormreluPW, 'conv5_2', [1,1,256,256])

                      .apply(convnormreluDW, 'conv6_1', [3,3,256,1], 1)
                      .apply(convnormreluPW, 'conv6_2', [1,1,256,256])
                      
                      .apply(convnormreluDW, 'conv7_1', [3,3,256,1], 1)
                      .apply(convnormreluPW, 'conv7_2', [1,1,256,256])

                      .apply(convnormreluDW, 'conv8_1', [3,3,256,1], 1)
                      .apply(convnormreluPW, 'conv8_2', [1,1,256,256])
                      
                      .apply(convnormreluDW, 'conv9_1', [3,3,256,1], 1)
                      .apply(convnormreluPW, 'conv9_2', [1,1,256,256])

                      .apply(convnormreluDW, 'conv10_1', [3,3,256,1], 2)
                      .apply(convnormreluPW, 'conv10_2', [1,1,256,1000])
                      
                      .apply(convnormreluDW, 'conv11_1', [3,3,1000,1], 1)
                      .Dropout('drop0', rate=1e-3)
                      .apply(convnormreluPW, 'conv11_2', [1,1,1000,1000])

                      .tf.reduce_mean(name='avg0', axis=[2,3])
		      .tf.reshape(name='shape0', shape=[-1,1000])())

     add_param_summary(('.*', ['histogram', 'rms']))
        return logits


def get_data(name, batch):
    isTrain = name == 'train'
    augmentors = fbresnet_augmentor(isTrain)
    return get_imagenet_dataflow(args.data, name, batch, augmentors)


def get_config():
    nr_tower = max(get_nr_gpu(), 1)
    batch = args.batch
    total_batch = batch * nr_tower
    assert total_batch >= 128   # otherwise the learning rate warmup is wrong.
    BASE_LR = 0.02 * (total_batch / 128.)
    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
    dataset_train = get_data('train', batch)
    dataset_val = get_data('val', batch)

    infs = [ClassificationError('wrong-top1', 'val-error-top1'),
            ClassificationError('wrong-top5', 'val-error-top5')]
    callbacks = [
        ModelSaver(),
        MinSaver('val-error-top1'),
        GPUUtilizationTracker(),
        EstimatedTimeLeft(),
        ScheduledHyperParamSetter(
            'learning_rate',
            [(0, 0.01), (3, max(BASE_LR, 0.01))], interp='linear'),
        ScheduledHyperParamSetter(
            'learning_rate',
            [(35, BASE_LR * 1e-1), (60, BASE_LR * 1e-2), (80, BASE_LR * 1e-3)]),
        DataParallelInferenceRunner(
            dataset_val, infs, list(range(nr_tower))),
    ]

    input = QueueInput(dataset_train)
    input = StagingInput(input, nr_stage=1)
    return TrainConfig(
        model=Model(),
        data=input,
        callbacks=callbacks,
        steps_per_epoch=1274490 // total_batch,
        max_epoch=110,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--batch', type=int, default=32, help='batch per GPU')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.set_logger_dir(os.path.join('train_log', 'hybridnet32_dropout_wo-max-v1'))

    config = get_config()
    if args.load:
        config.session_init = get_model_loader(args.load)
    nr_tower = max(get_nr_gpu(), 1)
    trainer = SyncMultiGPUTrainerReplicated(nr_tower)
    #trainer = SimpleTrainer()
    launch_train_with_config(config, trainer)

