import tensorflow as tf

def make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv(channels, stride=1, kernel_size=1, use_bias=False):
    return tf.keras.layers.Conv2D(filters=channels,
                                  kernel_size=(kernel_size, kernel_size),
                                  strides=stride,
                                  padding="same",
                                  use_bias=use_bias,
                                  )


def conv_dw(stride=1, kernel_size=3):
    return tf.keras.layers.DepthwiseConv2D(kernel_size=(kernel_size, kernel_size),
                                           strides=stride,
                                           padding="same",
                                           use_bias=False,
                                           )


def bn():
    return tf.keras.layers.BatchNormalization(momentum=0.9,
                                              epsilon=1e-3,
                                              )


def h_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6


# Tflite converter will automatically identify & replace hardswish in graph.  See:
#  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/toco/graph_transformations/identify_hardswish.cc
def h_swish(x):
    return x * tf.nn.relu6(x + 3) / 6


def pool(x):
    return tf.nn.avg_pool2d(x, ksize=(x.shape[1], x.shape[2]), padding='VALID', strides=(1, 1))


def get_act_fn(act_type, minimalistic=False, quantized=False):
    if minimalistic:
        if quantized:
            return tf.nn.relu6

        return tf.nn.relu
    if act_type == "HS":
        return h_swish
    elif act_type == "RE":
        if quantized:
            return tf.nn.relu6
        else:
            return tf.nn.relu
    else:
        raise NotImplementedError


def act(x, act_type, minimalistic=False, quantized=False):
    return get_act_fn(act_type, minimalistic=minimalistic, quantized=quantized)(x)


class SE(tf.keras.layers.Layer):
    def __init__(self, input_channels, r=4, quantized=False):
        super().__init__()
        self.input_channels = input_channels
        self.r = r
        self.quantized = quantized

        # Note: Official model uses bias!
        self.fc1 = conv(make_divisible(v=self.input_channels // self.r, divisor=8), use_bias=False)
        self.fc2 = conv(self.input_channels, use_bias=False)

    def get_config(self):
        config = {'input_channels': self.input_channels,
                  'r': self.r,
                  'quantized': self.quantized,
                  }
        base_config = super().get_config()
        return {**base_config, **config}

    def call(self, inputs, **kwargs):
        x = pool(inputs)
        x = self.fc1(x)
        if self.quantized:
            x = tf.nn.relu6(x)
        else:
            x = tf.nn.relu(x)

        x = self.fc2(x)
        x = h_sigmoid(x)

        x = tf.multiply(inputs, x)
        return x


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, in_size, exp_size, out_size, s, use_se, act_type, k, stem=False, minimalistic=False, quantized=False):
        super().__init__()
        self.in_size = in_size
        self.exp_size = exp_size
        self.out_size = out_size
        self.s = s
        self.use_se = use_se
        self.act_type = act_type
        self.k = k
        self.stem = stem
        self.minimalistic = minimalistic
        self.quantized = quantized

        # For minimalistic models, use ReLU and kernel size 3 everywhere & no SE block
        if self.minimalistic:
            if self.act_type == 'HS':
                self.act_type = 'RE'

            self.use_se = False
            self.k = 3

        kernel_size = 1
        stride = 1
        if self.stem:
            kernel_size = 3
            stride = 2

        self.conv1 = conv(self.exp_size, stride=stride, kernel_size=kernel_size)
        self.bn1 = bn()

        self.conv2 = conv_dw(stride=self.s, kernel_size=self.k)
        self.bn2 = bn()
        if self.use_se:
            self.se = SE(input_channels=self.exp_size, quantized=self.quantized)

        self.conv3 = conv(self.out_size)
        self.bn3 = bn()

    def get_config(self):
        config = {'in_size': self.in_size,
                  'exp_size': self.exp_size,
                  'out_size': self.out_size,
                  's': self.s,
                  'use_se': self.use_se,
                  'act_type': self.act_type,
                  'k': self.k,
                  'stem': self.stem,
                  'minimalistic': self.minimalistic,
                  'quantized': self.quantized,
                  }
        base_config = super().get_config()
        return {**base_config, **config}

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)

        if self.stem:
            if self.minimalistic:
                if self.quantized:
                    x = tf.nn.relu6(x)
                else:
                    x = tf.nn.relu(x)
            else:
                x = h_swish(x)
        else:
            x = act(x, self.act_type, self.minimalistic, self.quantized)

        if self.stem:
            inputs = x

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = act(x, self.act_type, self.minimalistic, self.quantized)

        if self.use_se:
            x = self.se(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        if self.s == 1 and self.in_size == self.out_size:
            x = tf.keras.layers.add([x, inputs])

        return x


def small_config():
    bottleneck_config = [
        {'in_size': 16, 'exp_size': 16, 'out_size': 16, 's': 2, 'k': 3, 'use_se': True, 'act_type': 'RE', 'stem': True},
        {'in_size': 16, 'exp_size': 72, 'out_size': 24, 's': 2, 'k': 3, 'use_se': False, 'act_type': 'RE'},
        {'in_size': 24, 'exp_size': 88, 'out_size': 24, 's': 1, 'k': 3, 'use_se': False, 'act_type': 'RE'},
        {'in_size': 24, 'exp_size': 96, 'out_size': 40, 's': 2, 'k': 5, 'use_se': True, 'act_type': 'HS'},
        {'in_size': 40, 'exp_size': 240, 'out_size': 40, 's': 1, 'k': 5, 'use_se': True, 'act_type': 'HS'},
        {'in_size': 40, 'exp_size': 240, 'out_size': 40, 's': 1, 'k': 5, 'use_se': True, 'act_type': 'HS'},
        {'in_size': 40, 'exp_size': 120, 'out_size': 48, 's': 1, 'k': 5, 'use_se': True, 'act_type': 'HS'},
        {'in_size': 48, 'exp_size': 144, 'out_size': 48, 's': 1, 'k': 5, 'use_se': True, 'act_type': 'HS'},
        {'in_size': 48, 'exp_size': 288, 'out_size': 96, 's': 2, 'k': 5, 'use_se': True, 'act_type': 'HS'},
        {'in_size': 96, 'exp_size': 576, 'out_size': 96, 's': 1, 'k': 5, 'use_se': True, 'act_type': 'HS'},
        {'in_size': 96, 'exp_size': 576, 'out_size': 96, 's': 1, 'k': 5, 'use_se': True, 'act_type': 'HS'},
    ]

    top_conv_filters = 576
    top_filters = 1024

    return bottleneck_config, top_conv_filters, top_filters


def large_config():
    bottleneck_config = [
        {'in_size': 16, 'exp_size': 16, 'out_size': 16, 's': 1, 'k': 3, 'use_se': False, 'act_type': 'RE', 'stem': True},
        {'in_size': 16, 'exp_size': 64, 'out_size': 24, 's': 2, 'k': 3, 'use_se': False, 'act_type': 'RE'},
        {'in_size': 24, 'exp_size': 72, 'out_size': 24, 's': 1, 'k': 3, 'use_se': False, 'act_type': 'RE'},
        {'in_size': 24, 'exp_size': 72, 'out_size': 40, 's': 2, 'k': 5, 'use_se': True, 'act_type': 'RE'},
        {'in_size': 40, 'exp_size': 120, 'out_size': 40, 's': 1, 'k': 5, 'use_se': True, 'act_type': 'RE'},
        {'in_size': 40, 'exp_size': 120, 'out_size': 40, 's': 1, 'k': 5, 'use_se': True, 'act_type': 'RE'},
        {'in_size': 40, 'exp_size': 240, 'out_size': 80, 's': 2, 'k': 3, 'use_se': False, 'act_type': 'HS'},
        {'in_size': 80, 'exp_size': 200, 'out_size': 80, 's': 1, 'k': 3, 'use_se': False, 'act_type': 'HS'},
        {'in_size': 80, 'exp_size': 184, 'out_size': 80, 's': 1, 'k': 3, 'use_se': False, 'act_type': 'HS'},
        {'in_size': 80, 'exp_size': 184, 'out_size': 80, 's': 1, 'k': 3, 'use_se': False, 'act_type': 'HS'},
        {'in_size': 80, 'exp_size': 480, 'out_size': 112, 's': 1, 'k': 3, 'use_se': True, 'act_type': 'HS'},
        {'in_size': 112, 'exp_size': 672, 'out_size': 112, 's': 1, 'k': 3, 'use_se': True, 'act_type': 'HS'},
        {'in_size': 112, 'exp_size': 672, 'out_size': 160, 's': 2, 'k': 5, 'use_se': True, 'act_type': 'HS'},
        {'in_size': 160, 'exp_size': 960, 'out_size': 160, 's': 1, 'k': 5, 'use_se': True, 'act_type': 'HS'},
        {'in_size': 160, 'exp_size': 960, 'out_size': 160, 's': 1, 'k': 5, 'use_se': True, 'act_type': 'HS'},
    ]

    top_conv_filters = 960
    top_filters = 1280

    return bottleneck_config, top_conv_filters, top_filters


class MobileNetV3(tf.keras.layers.Layer):
    def __init__(self, num_classes=1000, in_channels=3, minimalistic=False, softmax=True, mode='small', dropout=0.2, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.minimalistic = minimalistic
        self.softmax = softmax
        self.mode = mode
        self.dropout = dropout

        for kwarg in kwargs:
            print('Argument %s not understood' % kwarg)

        self.quantized = False
        if self.mode == 'small':
            bottleneck_config, top_conv_filters, top_filters = small_config()

        elif self.mode == 'small_quantized':
            self.quantized = True
            bottleneck_config, top_conv_filters, top_filters = small_config()

        elif self.mode == 'large':
            bottleneck_config, top_conv_filters, top_filters = large_config()

        elif self.mode == 'large_quantized':
            self.quantized = True
            bottleneck_config, top_conv_filters, top_filters = large_config()

        self.layers = []
        for config in bottleneck_config:
            self.layers.append(BottleNeck(**{**config, **{'quantized': self.quantized, 'minimalistic': self.minimalistic}}))

        self.conv1 = conv(top_conv_filters)
        self.bn1 = bn()
        self.conv2 = conv(top_filters, use_bias=True)
        self.conv3 = conv(self.num_classes, use_bias=True)

    def get_config(self):
        config = {'num_classes': self.num_classes,
                  'in_channels': self.in_channels,
                  'minimalistic': self.minimalistic,
                  'softmax': self.softmax,
                  'mode': self.mode,
                  'dropout': self.dropout,
                  }
        base_config = super().get_config()
        return {**base_config, **config}

    def call(self, x, training=False):
        for layer in self.layers:
            x = layer(x)

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = act(x, 'HS', minimalistic=self.minimalistic, quantized=self.quantized)
        x = pool(x)

        x = self.conv2(x)
        x = act(x, 'HS', minimalistic=self.minimalistic, quantized=self.quantized)

        # Need to use training flag here - dropout doesn't play nice with tflite converter
        if training:
            x = tf.nn.dropout(x, rate=self.dropout)

        x = self.conv3(x)
        x = tf.reshape(x, (-1, x.shape[-1]))

        if self.softmax:
            x = tf.nn.softmax(x)
        return x


# Need to use functional API to create model, so that it works with TPU
def mobilenet_v3(size=224, **kwargs):
    inputs = tf.keras.layers.Input(shape=(size, size, 3))
    output = MobileNetV3(**kwargs)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    return model


def mobilenet_v3_small(mode='small', pretrained=True, **kwargs):
    model = mobilenet_v3(mode=mode, **kwargs)
    if pretrained:
        print('Pretrained model specified - loading weights')
        model.load_weights('small_weights.h5')
    return model


def mobilenet_v3_small_minimalistic(mode='small', minimalistic=True, pretrained=True, **kwargs):
    model = mobilenet_v3(mode=mode, minimalistic=minimalistic, **kwargs)
    if pretrained:
        print('Pretrained model specified - loading weights')
        model.load_weights('small_minimalistic_weights.h5')
    return model


def mobilenet_v3_large(mode='large', **kwargs):
    return mobilenet_v3(mode=mode, **kwargs)


def mobilenet_v3_large_minimalistic(mode='large', minimalistic=True, **kwargs):
    return mobilenet_v3(mode=mode, minimalistic=minimalistic, **kwargs)
