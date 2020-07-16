import tensorflow as tf
from .residual_block import make_residual_block


layers = tf.keras.layers

class WideResNet(tf.keras.Model):

    def __init__(self, num_class, layer_params, l2_reg):
        super(WideResNet, self).__init__()
        num_filter1, num_filter2, num_filter3, num_block = layer_params
        self.first_layer_conv = layers.Conv2D(
            filters=16, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5,
            beta_regularizer=tf.keras.regularizers.l2(l=l2_reg), 
            gamma_regularizer=tf.keras.regularizers.l2(l=l2_reg))
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.layer1 = make_residual_block(num_filter1, num_block, l2_reg)
        self.layer2 = make_residual_block(
            num_filter2, num_block, l2_reg, downsample=True)
        self.layer3 = make_residual_block(
            num_filter3, num_block, l2_reg, downsample=True)
        self.fc = layers.Dense(num_class)

    def call(self, x, training=False):
        x = self.first_layer_conv(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.global_avg_pool(x)
        return self.fc(x)

def wide_resnet_28_2(num_class, l2_reg):
    return WideResNet(num_class, [32, 64, 128, 4], l2_reg)
