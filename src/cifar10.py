import tensorflow as tf
import numpy as np
from .data_aug import get_data_aug_fn

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_cifar10_data(batch_size, data_aug, train_data_size=None,
        repeat=True, shuffle=True, shuffle_size=None):
    train_data, test_data = tf.keras.datasets.cifar10.load_data()
    train_img, train_lbl = train_data[0].astype(np.float32), train_data[1]
    test_img, test_lbl = test_data[0].astype(np.float32), test_data[1]
    mean = np.array([125.307, 122.95, 113.865])
    std  = np.array([62.9932, 62.0887, 66.7048])
    train_img, test_img = (train_img - mean) / std, (test_img - mean) / std
    if train_data_size is not None:
        train_img = train_img[:train_data_size]
        train_lbl = train_lbl[:train_data_size]
    train_dataset = tf.data.Dataset.from_tensor_slices((train_img, train_lbl))
    if shuffle:
        assert shuffle_size is not None
        train_dataset = train_dataset.shuffle(shuffle_size)    
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    if data_aug:
        data_aug = get_data_aug_fn(batch_size, [4, 4, 32, 32, 3])
        train_dataset = train_dataset.map(data_aug, num_parallel_calls=AUTOTUNE)
    if repeat:
        train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.prefetch(AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_img, test_lbl))
    test_dataset = test_dataset.batch(100).prefetch(AUTOTUNE)
    return train_dataset, test_dataset