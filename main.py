import numpy as np
import mobilenet_v3
from tflite_utils import *
from imagenet_input import *
import time, argparse


def compile_validate_model(model, data, num_classes, val_batch_size, validation_steps, use_bfloat16=False, verbose=1):
    print('Compiling model for validation')

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(metrics=['categorical_accuracy',
                                                                        'top_k_categorical_accuracy'])
                  )

    print('Validating model')
    imagenet_val = ImageNetInput(is_training=False, data_dir=data, batch_size=val_batch_size,
                                 num_classes=num_classes, use_bfloat16=use_bfloat16)
    results = model.evaluate(imagenet_val.input_fn(), steps=validation_steps, verbose=verbose)
    time.sleep(1.)
    print('\n')
    print('Val loss %f, Val acc top-1: %f, top-5: %f' % tuple(results))


def to_np(a):
    if type(a) != np.ndarray:
        return a.numpy()
    return a


def test_tflite(model, val_iterator, iters, warmup=0, name='tflite'):
    print('Testing model..')
    predicted_tf = []

    for i in range(warmup):
        img, target = next(val_iterator)
        model(img)

    accuracies = []
    stt = time.time()
    for i in range(iters):
        try:
            img, target = next(val_iterator)
        except StopIteration:
            print('Dataset ran out of data early - i = %d' % i)
            break
        predicted_tf = model(img)
        predicted_tf = to_np(predicted_tf)

        accuracies.append(tf.keras.metrics.categorical_accuracy(target, predicted_tf))

    total_time = time.time() - stt
    print('%s: Total time for %d images was %0.2f secs, which is %0.2f images/s' % (name, iters, total_time, iters / total_time))

    print('Output mean is: %f, std dev is: %f' % (predicted_tf.mean(), predicted_tf.std()))

    accuracies = 100. * tf.reduce_mean(accuracies)
    print('%s accuracy is: %0.2f' % (name, accuracies))


parser = argparse.ArgumentParser(description='Tensorflow 2.0 Tflite conversion')
parser.add_argument('--data_dir', metavar='DIR', help='Path to testing data.')
parser.add_argument('--arch', help='Architecture to test & convert to tensorflow lite.')
parser.add_argument('--num_steps', default=5000, type=int, help='number of batches to validate the model with')
parser.add_argument('--num_classes', default=1000, type=int, help='number of classes in dataset')
parser.add_argument('--batch_size', default=10, type=int, metavar='N', help='mini-batch size')
args = parser.parse_args()

model = mobilenet_v3.__dict__[args.arch](num_classes=args.num_classes)

# Validate model before tflite conversion
compile_validate_model(model, args.data_dir, args.num_classes, args.batch_size, args.num_steps)

# Create dataset & convert to tensorflow lite
imagenet_val = ImageNetInput(is_training=False, data_dir=args.data_dir, batch_size=1, num_classes=args.num_classes)
model_tflite = convert_keras_to_tflite(model, output_path='model.tflite')

# Check tflite model accuracy.  Note that we always uses batch size 1
test_tflite(LiteWrapper(model_tflite), iter(imagenet_val.input_fn()), args.num_steps * args.batch_size, 0)
