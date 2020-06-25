import tensorflow as tf


layers = tf.keras.layers

class ResidualBlock(layers.Layer):

    def __init__(self, num_filter, l2_reg, downsample=False):
        super(ResidualBlock, self).__init__()
        self.num_filter = num_filter
        stride = 2 if downsample else 1
        self.conv1 = layers.Conv2D(num_filter, 3, strides=stride, 
            kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg),
            padding="same", use_bias=False)
        self.conv2 = layers.Conv2D(num_filter, 3, strides=1,
            kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg),
            padding="same", use_bias=False)
        self.downsample_conv = layers.Conv2D(
            num_filter, 1, strides=stride, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5,
            beta_regularizer=tf.keras.regularizers.l2(l=l2_reg), 
            gamma_regularizer=tf.keras.regularizers.l2(l=l2_reg))
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5,
            beta_regularizer=tf.keras.regularizers.l2(l=l2_reg), 
            gamma_regularizer=tf.keras.regularizers.l2(l=l2_reg))

    def call(self, x, training=None):
        residual = x
        if self.num_filter != residual.shape[-1]: # Assuming NHWC data format.
            residual = self.downsample_conv(residual)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        return x + residual

def make_residual_block(num_filter, num_block, l2_reg, downsample=False):
    block = tf.keras.Sequential()
    block.add(ResidualBlock(num_filter, l2_reg, downsample=downsample))
    for _ in range(1, num_block):
        block.add(ResidualBlock(num_filter, l2_reg))
    return block