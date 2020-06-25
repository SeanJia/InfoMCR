# import tensorflow as tf
# import numpy as np

# jacobian = tf.convert_to_tensor(np.random.randn(10, 1000000))

# s = tf.linalg.svd(jacobian, compute_uv=False)

import tensorflow as tf
import numpy as np


# a = np.random.randn(100, 1000000)

# @tf.function
# def t():
#     return tf.linalg.svd(a, compute_uv=False)
# a = t()

x = tf.constant([5.0, np.inf, 6.8, np.inf])
tf.math.is_inf(x)
print(tf.reduce_mean(x))


# tf.summary.scalar('learning_rate', optimizer._decayed_lr(tf.float32), step=step) 
# 1. large scale parallel
# 2. diff across batch (not across forward step)
# 3. across forward step approximation (like Nestorove momentum)
# 4. adaptive regularizer coefficient
# 5. adaptive reg (dimension-wise)

# @tf.function
# def train_loss_eval(images, labels):
#     predictions = model(images, training=True)
#     predictions /= tf.expand_dims(tf.linalg.norm(predictions, axis=-1), 1) *
#     loss = loss_fn(y_true=labels, y_pred=scale * predictions)
#     train_loss(loss)
#     loss_fn2 = tf.nn.sparse_softmax_cross_entropy_with_logits
#     loss = loss_fn2(labels=tf.cast(
#         tf.squeeze(labels), tf.int32), logits=scale * predictions)
