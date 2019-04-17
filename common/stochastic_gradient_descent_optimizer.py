"""
stochastic gradient descent optimizer with exponential learning rate decay
"""
import tensorflow as tf


class StochasticGradientDescentOptimizer(object):
    def __init__(
            self,
            optim_params
    ):
        self.steps = tf.Variable(
            0,
            dtype=tf.int32,
            trainable=False,
            name="steps"
        )
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=optim_params["learning_rate"],
            global_step=self.steps,
            decay_steps=optim_params["decay_steps"],
            decay_rate=optim_params["decay_rate"],
            name="learning_rate"
        )
        self._sgd_optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate
        )
        self.sgd_optimizer = None
        self.optimizer = None

    def minimize(self, loss):
        self.sgd_optimizer = self._sgd_optimizer.minimize(
            loss=loss
        )
        with tf.control_dependencies(
            control_inputs=[self.sgd_optimizer]
        ):
            steps_new = self.steps + 1
            op_step_update = self.steps.assign(
                steps_new
            )
        self.optimizer = tf.group(
            self.sgd_optimizer, op_step_update
        )
        return self.optimizer
