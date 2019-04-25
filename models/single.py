"""
single model for all tasks
"""
import tensorflow as tf

from models import SharedBottom


class FC(SharedBottom):
    def __init__(
            self,
            **kwargs
    ):
        super(FC, self).__init__(**kwargs)

    def _setup_task_specific_top(
            self,
            feature,
            scope="task_specific_top"
    ):
        logits_, weights, biases, saver = self._setup_task_specific_block(
            feature=feature,
            task_dim=1,
            model_spec=self.model_spec,
            out_dim=self.label_dim,
            scope=scope
        )
        # to be consistent with final task projection in SharedBottom._setup_net #
        logits = tf.tile(
            input=logits_,
            multiples=[1, self.task_dim, 1]
        )
        return logits, weights, biases, saver
