import tensorflow as tf
from tensorflow import TensorSpec
from tensorflow.data import Dataset, AUTOTUNE
import numpy as np


class SimpleDataset:
    def __init__(self, a=1, b=2, c=0.1, noise_std=0.1):
        # create an ax + by + c > 0 model
        self.w = tf.convert_to_tensor([[a], [b]])
        self.b = c

        # with random noise
        self.noise_std = noise_std

    def generate(self):
        while True:
            # generate input between -1 and 1
            x = 2 * tf.random.uniform([1, 2], dtype=tf.float32) - 1

            # generate output
            y = tf.matmul(x, self.w) + self.b
            # create binary label for input
            y = tf.cast(y > 0, dtype=tf.float32) * 2 - 1

            # add noise to inputs
            x += tf.random.normal([1, 2], 0, self.noise_std, dtype=tf.float32)

            yield x[0], y[0]

    def tfdataset(self, batch_size: int = 1):
        # create dataset from generator
        dataset = Dataset.from_generator(
            self.generate,
            output_signature=(
                TensorSpec(
                    shape=(2),
                    dtype=tf.float32,
                ),
                TensorSpec(
                    shape=(1),
                    dtype=tf.float32,
                ),
            ),
        )

        # prefetch for performance
        dataset = dataset.prefetch(AUTOTUNE)

        # batch dataset
        dataset = dataset.batch(batch_size)

        return dataset
