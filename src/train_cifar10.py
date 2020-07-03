import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from wide_resnet import wide_resnet_28_2
from data import get_cifar10_data
from absl import app
from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_float('l2_reg', 5e-4, 'Coefficient of the l2 regularization.') 
flags.DEFINE_string('gpu_id', '0', 'GPU ID.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('train_data_size', 50000, 'Number of input data.')
flags.DEFINE_integer('num_class', 10, 'Number of classes in input data.')
flags.DEFINE_string('model_id', None, 'Model string ID.')
flags.DEFINE_integer('num_epoch', 200, 'Number of epochs in training.')
flags.DEFINE_integer('test_every', 500, 'How often does it test.')
flags.DEFINE_integer('save_every', 10000, 'How often does it save checkpoints.')
flags.DEFINE_boolean('data_aug', True, 'Whether augment data in training.')
flags.DEFINE_integer('num_sub_batch', 8, 'Number of sub-batch M.')
flags.DEFINE_float('alpha', 1e-4, 'Coefficient for linear approaximation.')
flags.DEFINE_float('beta', 10.0, 'Coefficient for local min reg.')
flags.DEFINE_integer('type', 1, 'Type of regularizer variants.')
flags.DEFINE_boolean('use_local_min_reg',
                     False, 'Whether enable the proposed regularizer.')
MODEL_BASE = 'models'


def get_model(num_class, l2_reg):
    model = wide_resnet_28_2(num_class, l2_reg)
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    return model

def learning_rate_schedule(it_per_ep):
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [int(60 * it_per_ep), int(120 * it_per_ep), int(160 * it_per_ep)], 
        [0.1, 0.1 * 0.2, 0.1 * 0.2 ** 2, 0.1 * 0.2 ** 3]
    )

def get_summary_writer(path): 
    train_log_dir, test_log_dir = f'{path}/train', f'{path}/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    return train_summary_writer, test_summary_writer

def obtain_loss_and_grad(model, img, lbl, loss_fn):
    with tf.GradientTape() as tape:
        pred = model(img, training=True)
        loss = loss_fn(y_true=lbl, y_pred=pred)
    return loss, tape.gradient(loss, model.trainable_variables)

def use_reg(step, type_idx, it_per_ep):
    type_string = ['111', '011', '001', '100', '010'][type_idx]
    if step < int(60 * it_per_ep):
        return type_string[0] == '1'
    if int(60 * it_per_ep) <= step < int(120 * it_per_ep):
        return type_string[1] == '1'
    if step >= int(120 * it_per_ep):
        return type_string[2] == '1'

def main(args):
    assert FLAGS.model_id is not None, 'Should specify --model_id'
    model_path = os.path.join(MODEL_BASE, FLAGS.model_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id
    
    train_data, test_data = get_cifar10_data(
        FLAGS.batch_size, FLAGS.data_aug, shuffle_size=50000)
    train_data = iter(train_data)
    model = get_model(FLAGS.num_class, FLAGS.l2_reg)
    print(f'PATH: {model_path}')
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
    iter_per_epoch = FLAGS.train_data_size / FLAGS.batch_size
    learning_rate_fn = learning_rate_schedule(iter_per_epoch)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate_fn, momentum=0.9, nesterov=True)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    train_l2 = tf.keras.metrics.Mean(name='train_l2')
    train_summary_writer, test_summary_writer = get_summary_writer(model_path)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_prefix = os.path.join(model_path, 'checkpoints', 'ckpt')

    model_copy = get_model(FLAGS.num_class, 0.0)
    trainable_vars = {v.name: idx
        for idx, v in enumerate(model.trainable_variables)}
    train_reg = tf.keras.metrics.Mean(name='train_reg')

    def get_regularizer_and_grad(images, labels, M, alpha):
        regs, grads = [], [[] for v in model.trainable_variables]
        for img, lbl in zip(tf.split(images, M), tf.split(labels, M)):
            # Reset model_copy.
            for i, w in enumerate(model.weights):
                model_copy.weights[i].assign(w)
            loss1, grad1 = obtain_loss_and_grad(model_copy, img, lbl, loss_fn)
            # Set up weights for model_copy.
            for i, w in enumerate(model.weights):
                if w.name in trainable_vars:
                    g = tf.stop_gradient(grad1[trainable_vars[w.name]])
                    model_copy.weights[i].assign(w - alpha * g)
                else:
                    model_copy.weights[i].assign(w)
            loss2, grad2 = obtain_loss_and_grad(model_copy, img, lbl, loss_fn)
            regs.append(loss1 - loss2)
            for i, (g1, g2) in enumerate(zip(grad1, grad2)):
                grads[i].append(g1 - g2)
        return tf.reduce_mean(regs), [tf.reduce_mean(gs, 0) for gs in grads]

    def get_beta(step):
        return FLAGS.beta if step > int(0 * iter_per_epoch) else 0.0

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(y_true=labels, y_pred=predictions)
            l2_reg = tf.add_n(model.losses)
            total_loss = loss + l2_reg
        grad = tape.gradient(total_loss, model.trainable_variables)         
        train_loss(loss)
        train_accuracy(labels, predictions)
        train_l2(l2_reg)
        optimizer.apply_gradients(
            grads_and_vars=zip(grad, model.trainable_variables))

    @tf.function
    def regularized_train_step(images, labels, beta):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(y_true=labels, y_pred=predictions)
            l2_reg = tf.add_n(model.losses)
            total_loss = loss + l2_reg
        grad1 = tape.gradient(total_loss, model.trainable_variables)
        local_min_reg, grad2 = get_regularizer_and_grad(
            images, labels, FLAGS.num_sub_batch, FLAGS.alpha)
        grad = [g1 + beta * g2 for g1, g2 in zip(grad1, grad2)]
        train_loss(loss)
        train_accuracy(labels, predictions)
        train_l2(l2_reg)
        train_reg(local_min_reg)
        optimizer.apply_gradients(
            grads_and_vars=zip(grad, model.trainable_variables))

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        loss = loss_fn(labels, predictions)
        test_loss(loss)
        test_accuracy(labels, predictions)

    for step in tqdm(range(int(FLAGS.num_epoch * iter_per_epoch))):
        if FLAGS.use_local_min_reg and use_reg(step, FLAGS.type, iter_per_epoch):
            regularized_train_step(*next(train_data), FLAGS.beta)
        else:
            train_step(*next(train_data))
        if step % FLAGS.test_every == 0:
            for images, labels in test_data:
                test_step(images, labels)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=step)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=step) 
                tf.summary.scalar('l2_reg', train_l2.result(), step=step) 
                tf.summary.scalar('local_min_reg', train_reg.result(), step=step) 
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=step)
                tf.summary.scalar('accuracy', test_accuracy.result(), step=step)
            train_loss.reset_states()
            train_accuracy.reset_states()
            train_l2.reset_states()
            train_reg.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()
        if step % FLAGS.save_every == 0 and step > 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

if __name__ == '__main__':
    app.run(main)
