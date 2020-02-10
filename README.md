# MobileNet V3 Inc Minimalistic - Tensorflow 2.0 & Lite

A MobileNet V3 implementation in Tensorflow 2.0, with Tensorflow Lite (tflite) conversion & benchmarks.  I created this repo as there isn't an official Keras/TF 2.0 implementation of MobileNet V3 yet and all the existing repos I looked at didn't contain minimalistic or tflite implementations, including how to use the accelerated hardswish operation provided in tflite nor profilings.

Additionally all the repos I looked at contain differences to the official MobileNet V3 implementation [here](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet).  This is probably due to following early versions of the paper.  Some mistakes I found:

* The stem convolution is included into the first bottleneck block
* The Squeeze-Excite module is after the depthwise activation, not before.  This is important, otherwise conv & relu operations won't be fused in the tflite model
* The number of squeeze channels should be rounded up to the nearest multiple of 8
* The small model uses 1024 channels in the pooled head, not 1280

This implementation was verified by comparing the converted tflite models against the official implementation.  That being said, this implementation doesn't use biases on the Squeeze-Excite blocks or the redundant average pool op in the head of the reference models and includes a softmax layer on the output.

Some further notes on my implementation & tflite:

* As dropout doesn't play nice with tflite converter, the dropout layer is only applied during training 
* All dense layers are implemented as 4D convolutions.  This removes the need for additional reshape ops in the tflite graph
* The tflite converter automatically identifies & replaces the hardswish operation, but only on recent versions of tensorflow
* I use the experimental MLIR based converter

Note: I strongly advise building Tensorflow from source, in order for the following to function correctly:

* Hardswish replacement
* MLIR converter
* XNNPACK
* Performance: Until recently tflite on linux only used O2 instead of O3 optimisation

# Benchmarks
After tweaking the training process a bit, I was able to meet or exceed the reference accuracy for the small models:

Network             | Official Top-1 Accuracy | Top-1 Accuracy
---                 | ---                     | ---
Small               | 67.5                    | 67.6
Small Minimalistic  | 61.9                    | 63.5

Weights are included in the repo.

Here are some benchmarks, including with the new XNNPACK backend for Tensorflow Lite:

Device                                  | Small (ms) | Small Minimalistic (ms) | Large (ms) | Large Minimalistic (ms)
---                                     | ---        | ---                     | ---        | ---
ODroid N2                               | 22.1       | 17.6                    | 70.4       | 62.1
Samsung Galaxy S8, CPU	                 | 17         | 13.8                    | 93         | 44.8
Samsung Galaxy S8, CPU, XNNPACK backend | 11.7       | 8.65                    | 36.7       | 31.9
Samsung Galaxy S8, GPU backend          | 13         | 5.16                    | 12.7       | 11.3

I tested on 1 core over 1000 iterations, with 50 warmup iterations.

# Usage Instructions

The dataloader is taken from the ResNet50 Tensorflow example dataloader [here](https://github.com/tensorflow/tpu/blob/master/models/experimental/resnet50_keras/imagenet_input.py). You can prep the tfrecord files using the script [here.](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py)

You can execute the script like:

```
python main.py --data_dir <path/to/data> --arch mobilenet_v3_small
```

Benchmarks were done using the script [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)


