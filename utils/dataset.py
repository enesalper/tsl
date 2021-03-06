import os
from functools import partial

import numpy as np
import tensorflow as tf


from utils.augmentation import train_transforms


def decode_image(path, labels):
    img_bin = tf.io.read_file(path)
    label = tf.strings.split(path, "/")[-2]
    ext = tf.strings.split(path, ".")[-1]

    if ext == "png":
        img = tf.image.decode_png(img_bin, channels=3)

    elif ext == "jpeg" or ext == "jpg":
        img = tf.image.decode_jpeg(img_bin, channels=3)

    else:
        img = tf.image.decode_png(img_bin, channels=3)

    encoded_label = tf.cast(label == labels, dtype=tf.float32)
    return img, encoded_label


def process_image(image, label_encoded, target_size=(224,224), is_training=False):
    img = tf.image.resize(image, target_size)
    img = aug_process(img, is_training)
    img = img / 255.
    return img, label_encoded


def aug_fn(image, is_training):
    if is_training:
        aug_data = train_transforms(image=image)
        image = aug_data["image"]
    return image


def aug_process(image, is_training):
    aug_img = tf.numpy_function(func=aug_fn, inp=[image, is_training], Tout=tf.float32)
    return aug_img


class Dataset:
    def __init__(self, path: str, extension_type: str, num_parallel_calls: int, random_seed=1,
                 is_training=False, target_shape=(224,224)):
        self.path = path
        self.extension_type = extension_type
        self.num_parallel_calls = num_parallel_calls
        self.random_seed = random_seed
        self.is_training = is_training
        self.target_shape = target_shape

        if random_seed > 0:
            tf.random.set_seed(self.random_seed)
            self.shuffle = True
        else:
            self.shuffle = False

        self.labels = sorted(tf.io.gfile.listdir(self.path))
        self.ds = self.read_files()

        self.decode_ds = self.map_files()

    def read_files(self) -> tf.data.Dataset:
        files = tf.io.gfile.glob(os.path.join(self.path, f'**/*.{self.extension_type}'))

        if self.shuffle:
            files = tf.random.shuffle(files)
        else:
            files = tf.convert_to_tensor(files)
        return tf.data.Dataset.from_tensor_slices(files)

    def map_files(self):
        decode_ds = self.ds.map(partial(decode_image, labels=self.labels), self.num_parallel_calls)
        self.ds = decode_ds.map(partial(process_image, target_size=self.target_shape, is_training=self.is_training), self.num_parallel_calls)
        return decode_ds

    def get_ds(self):
        return self.ds

    def get_decode_ds(self):
        return self.decode_ds


if __name__ == "__main__":
    dataset_train = Dataset("../data/Train", "png", num_parallel_calls=tf.data.experimental.AUTOTUNE, is_training=True, target_shape= (224,224))
    train_ds = dataset_train.get_ds()

    dataset_test = Dataset("../data/Test", "png", num_parallel_calls=tf.data.experimental.AUTOTUNE, is_training=False, target_shape= (224,224))
    test_ds = dataset_test.get_ds()

    for im, l in train_ds.take(10).as_numpy_iterator():
        print(np.min(im))
        print(np.max(im))

