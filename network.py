import ops
import config
import numpy as np
import tensorflow as tf

class net:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, train_phase):
        with tf.variable_scope(self.name):
            inputs = tf.nn.relu(ops.conv("conv0", inputs, 64, 3, 1))
            for d in np.arange(1, config.DEPTH - 1):
                inputs = tf.nn.relu(ops.batchnorm(ops.conv("conv_" + str(d + 1), inputs, 64, 3, 1), train_phase, "bn" + str(d)))
            inputs = ops.conv("conv" + str(config.DEPTH - 1), inputs, config.IMG_C, 3, 1)
            return inputs