import errno, os
import numpy as np
import tensorflow as tf

def mkdir_p(path):
    """make dir if not exist"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_filename(path):
    return os.path.basename(os.path.normpath(path))


class LiteWrapper():
    """
    A simple wrapper on a tflite model to allow TF 2.0 eager/pytorch style execution
    """
    def __init__(self, model):
        self.model = model
        self.input_details = model.get_input_details()
        self.output_details = model.get_output_details()

    def __call__(self, x):
        if self.input_details[0]['dtype'] == np.uint8 and x.dtype != np.uint8:
            print('Casting input to uint8')
            x = tf.cast(x, tf.uint8)
        self.model.set_tensor(self.input_details[0]['index'], x)
        self.model.invoke()
        results = self.model.get_tensor(self.output_details[0]['index'])
        return results


# NOTE: tf 2.0 doesn't support quantized inputs/outputs ATM..
# https://github.com/tensorflow/tensorflow/issues/34416
def convert_keras_to_tflite(model,
                            output_path=None,
                            quantize=False,
                            quantize_weights=False,  # Weight quantization only
                            calibration_examples=1000,
                            iterator=None):
    """
    This function converts a TF 2.0 keras model to tensorflow lite.  It includes options for weight quantisation &
    post training quantisationn.
    :param model: Input model to quantise
    :param output_path: Filename to save output tflite model to
    :param quantize: Quantise the tflite model (activations & weights)
    :param quantize_weights: Quantise the tflite model (weights only, for filesize reduction)
    :param calibration_examples: Number of calibration examples to use to determine quantisationn ranges
    :param iterator: Representative dataset to determine quantisation ranges
    :return: tflite interpreter holding the converted model
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize_weights:
        assert quantize is False, 'Only full quantization or weight quantization can be selected'
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantize:
        assert quantize_weights is False, 'Only full quantization or weight quantization can be selected'
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        assert iterator is not None, 'For post-training integer quantization, a representative dataset is needed ' \
                                     'to calibrate the min/max values of every tensor'

        def representative_data_gen(num_examples=calibration_examples):
            for i in range(num_examples):
                data = next(iterator)[0]
                yield [data]

        converter.representative_dataset = representative_data_gen

    converter.experimental_enable_mlir_converter = True  # This works for float models - not quantisation
    tflite_model = converter.convert()

    if output_path is not None:
        print('Writing tensorflow lite model to: %s' % output_path)
        open(output_path, "wb").write(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    return interpreter
