import numpy as np
import tensorflow as tf

from utils.dataset import Dataset


input_shape = (224, 224, 3)

batch_size = 128

dataset_test = Dataset('data/Test', 'png', num_parallel_calls=tf.data.experimental.AUTOTUNE,
                       is_training=False, target_shape=input_shape[:2])

test_ds = dataset_test.get_ds()
test_ds = test_ds.cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

m = tf.keras.models.load_model("./saved_models/TSL_MobileNetV2.h5")

