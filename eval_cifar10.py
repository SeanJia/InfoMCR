import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from wide_resnet import wide_resnet_28_2
from data import get_cifar10_data
from absl import app
from absl import flags


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

FLAGS = flags.FLAGS
flags.DEFINE_string('gpu_id', '0', 'GPU ID.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('train_data_size', 10000, 'Number of input data.')
flags.DEFINE_integer('num_class', 10, 'Number of classes in input data.')
flags.DEFINE_float('scale', 0.04, 'Normalization scale for computing gamma.')
flags.DEFINE_integer('chunk_size', 10, 'chunk_size * num_chunk = batch_size.')
flags.DEFINE_integer('num_chunk', 10, 'chunk_size * num_chunk = batch_size.')
flags.DEFINE_integer('T', 100, 'Number of trials for computing gamma.')
flags.DEFINE_string('model_id', None, 'Model string ID.')
MODEL_BASE = 'models'

def get_model(num_class):
    model = wide_resnet_28_2(num_class, 0.0)
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    return model

def compute_logsum(model, images, labels, scale, chunk_size, num_chunk):
    assert images.shape[0] == chunk_size * num_chunk
    labels = tf.cast(tf.squeeze(labels), tf.int32)
    images, labels = tf.split(images, num_chunk), tf.split(labels, num_chunk)
    loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    @tf.function
    def get_jacobian(imgs, lbls):
        with tf.GradientTape() as tape:
            predictions = model(imgs, training=True)
            predictions /= tf.expand_dims(tf.linalg.norm(predictions, axis=-1), 1) * scale
            loss = loss_fn(labels=lbls, logits=predictions)
        jacobian = tape.jacobian(loss, model.trainable_variables)
        jacobian = [tf.split(tf.reshape(j, [-1]), chunk_size) for j in jacobian]
        jacobian_temp = [[] for _ in range(chunk_size)]
        for jac in jacobian:
            for i in range(chunk_size):
                jacobian_temp[i].append(jac[i])
        jacobian = tf.stack([tf.concat(jac, 0) for jac in jacobian_temp])
        return jacobian

    jacobian = tf.concat(
        [get_jacobian(images[i], labels[i]) for i in range(num_chunk)], 0)

    @tf.function
    def get_logsum():
        s = tf.linalg.svd(jacobian.numpy().astype(np.float64), compute_uv=False)
        return tf.reduce_sum(tf.math.log(s))
    
    return get_logsum()

def compute_gamma(model, data, scale, chunk_size=10, num_chunk=10, T=100):
    gamma = []
    i = 0
    for i, (imgs, lbls) in tqdm(enumerate(data)):
        if i == T:
            break
        logsum = compute_logsum(model, imgs, lbls, scale, chunk_size, num_chunk)
        tf.print(logsum)
        gamma.append(logsum)
    return np.mean(gamma)
        

def main(args):
    assert FLAGS.model_id is not None, 'Should specify --model_id'
    model_path = os.path.join(MODEL_BASE, FLAGS.model_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id
    
    train_data, test_data = get_cifar10_data(FLAGS.batch_size,
        False, FLAGS.train_data_size, False, False)
    train_data = iter(train_data)
    model = get_model(FLAGS.num_class)
    print(f'PATH: {model_path}')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_path = os.path.join(model_path, 'checkpoints')
    checkpoint.restore(
        tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

    @tf.function
    def test_acc_eval(images, labels):
        predictions = model(images, training=False)
        test_accuracy(labels, predictions)

    for images, labels in test_data: 
        test_acc_eval(images, labels)
    acc = test_accuracy.result()
    print(f'The test accuracy is {acc.numpy() * 100} %.')

    gamma = compute_gamma(
        model, train_data, FLAGS.scale, chunk_size=10, num_chunk=10, T=100)
    print(f'The compuated gamm is {gamma}.')


if __name__ == '__main__':
    app.run(main)
