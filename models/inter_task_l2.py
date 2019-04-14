"""
inter-task l2 regularization to enforce inter-task relationship
@inproceedings{duong2015low,
  title={Low resource dependency parsing: Cross-lingual parameter sharing in a neural network parser},
  author={Duong, Long and Cohn, Trevor and Bird, Steven and Cook, Paul},
  booktitle={Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)},
  volume={2},
  pages={845--850},
  year={2015}
}
extended from \cite{duong2015low},
    1. where there are only two tasks, here for multiple tasks, exhausted pair-wise loss is used
    2. shared-bottom structure is used to enable minimum comparison
"""
import tensorflow as tf

from models import SharedBottom


class InterTaskL2(SharedBottom):
    def __init__(
            self,
            **kwargs
    ):
        super(InterTaskL2, self).__init__(**kwargs)

    def _setup_regularization(self):
        regularization_loss = super(InterTaskL2, self)._setup_regularization()

        task_pairwise_l2 = 0.0
        for weight in self.task_weight_list + self.task_bias_list:
            weight_pair_diff = tf.expand_dims(weight, axis=1) - tf.expand_dims(weight, axis=0)
            task_pairwise_l2 += tf.pow(tf.norm(
                weight_pair_diff,
                ord='euclidean'
            ), 2.0)
            # task_pairwise_l2 += task_pairwise_l2_single
        regularization_loss = regularization_loss + self.model_spec["task_pairwise_l2"] * task_pairwise_l2
        return regularization_loss
