import tensorflow as tf
from tensorflow.keras.layers import Dense

from tqdm import tqdm
import wandb


class SimpleModel:
    def __init__(self, batch_size, learning_rate, num_gpus, strategy=None):
        # create a simple classification model
        self.model = Dense(1, use_bias=True)

        # mse loss
        self.loss_function = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM
        )

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # global batch size
        self.batch_size = batch_size
        self.global_batch_size = batch_size * num_gpus

        # strategy to distribute training
        self.strategy = strategy

    @tf.function
    def train_step(self, x, y):
        """
        Take a training step given a batch of data x and label y.

        :param x: input data batch
        :type x: tf.Tensor
        :param y: corresponding labels
        :type y: tf.Tensor
        :return: loss
        :rtype: tf.Tensor
        """
        with tf.GradientTape(persistent=True) as tape:
            # generate predictions
            y_hat = self.model(x)
            # calculate loss
            loss = self.loss_function(y / 2 + 0.5, y_hat) / self.batch_size

        # calculate gradient
        grad = tape.gradient(loss, self.model.trainable_variables)

        # optimize
        self.optimizer.apply_gradients(
            zip(grad, self.model.trainable_variables)
        )

        return loss

    @tf.function
    def distributed_train_step(self, x, y):
        """
        Training step for distributed (multi-gpu) training.

        :param x: input data batch
        :type x: tf.Tensor
        :param y: corresponding labels
        :type y: tf.Tensor
        :return: loss
        :rtype: tf.Tensor
        """
        per_replica_losses = self.strategy.run(self.train_step, args=(x, y))
        return self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None
        )

    def train(
        self,
        dataset,
        steps: int,
        log_save_step: int,
    ):
        """
        Train a model on the provided dataset.

        :param dataset: Tensorflow Dataset to train on
        :type dataset: tf.data.Dataset
        :param train_steps: Number of examples to train on (batch_size * iterations)
        :type train_steps: int
        :param log_save_step: Number of examples between logging
        :type log_save_step: int
        """
        assert (
            log_save_step % self.global_batch_size == 0
        ), "log save step must be divisible by global batch size"
        assert (
            steps % self.global_batch_size == 0
        ), "training steps must be divisible by global batch size"

        if self.strategy:
            dataset = self.strategy.experimental_distribute_dataset(dataset)

        train_steps = 0

        while train_steps < steps:
            pbar = tqdm(dataset)
            for x, y in pbar:
                # stop training after train_steps exceeds steps
                if train_steps >= steps:
                    break

                # Train on input batch
                if self.strategy:
                    loss = self.distributed_train_step(x, y).numpy().mean()
                else:
                    loss = self.train_step(x, y).numpy().mean()

                wandb.log(data={"BCE Loss": loss}, step=train_steps)
                pbar.set_postfix_str(f"BCE Loss: {loss}")

                if train_steps % log_save_step == 0:
                    # extract weights ax + by + c > 0
                    if self.strategy:
                        weights = self.model.weights[0].values[0]
                        bias = self.model.bias.values[0]
                    else:
                        weights = self.model.weights[0]
                        bias = self.model.bias[0]

                    a = weights[0].numpy()
                    b = weights[1].numpy()
                    c = bias.numpy()

                    wandb.log(
                        {"a coef": a, "b coef": b, "c coef": c},
                        step=train_steps,
                    )

                train_steps += self.global_batch_size
