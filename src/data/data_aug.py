import tensorflow as tf


def get_data_aug_fn(batch_size, aug_params):
    h_off, w_off, h, w, c = aug_params
    def data_aug(images, *args):
        images = tf.image.random_flip_left_right(images)
        images = tf.image.pad_to_bounding_box(
            images, h_off, w_off, h_off * 2 + h, w_off * 2 + w) 
        images = tf.image.random_crop(images, [batch_size, h, w, c])
        return tuple([images] + list(args))
    return data_aug
