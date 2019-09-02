"""
topic task sparse network with exclusive sparsity regularization
"""
import tensorflow as tf

from models import TopicTaskSparseLayerWise


class TopicTaskSparseLayerWiseExclusive(TopicTaskSparseLayerWise):
    @staticmethod
    def proximal_operator(
            weight,
            **kwargs
    ):
        return proximal_operator_exclusive_sparse(
            weight=weight,
            **kwargs
        )


def proximal_operator_exclusive_sparse(
        weight,
        eta,
        epsilon,
        **kwargs
):
    """
    calculate proximal operator for exclusive sparse regularization (2, 1, 2) norm
    :param weight: shape=[topic_dim, task_dim, inner_dim]
    :param eta: learning_rate * regularization_lambda
    :param epsilon: stabilisation term
    :param kwargs:
    :return:
    """
    weight_inner_norm = tf.norm(
        weight,
        ord='euclidean',
        axis=-1,
        keepdims=True
    )
    threshold = find_threshold_biased_partial_mean(
        weight_inner_norm=weight_inner_norm,
        eta=eta
    )
    weight_inner_norm_stabled = weight_inner_norm + epsilon
    weight_normalized = weight / weight_inner_norm_stabled
    scale = tf.nn.relu(
        weight_inner_norm - threshold
    )               # rather than using stabled version to keep sparsity correct
    weight_new = scale * weight_normalized
    return weight_new


def find_threshold_biased_partial_mean(
        weight_inner_norm,
        eta
):
    # find threshold #
    weight_inner_norm_sorted = tf.sort(
        weight_inner_norm,
        axis=-2,
        direction='DESCENDING'
    )
    cum_sums = tf.cumsum(
        weight_inner_norm_sorted,
        axis=-2,
        exclusive=False
    )
    counts = tf.range(
        start=1.0,
        limit=tf.cast(cum_sums.shape[-2].value, tf.float32) + 1.0,
        delta=1.0,
        dtype=tf.float32
    )
    counts_effective_inv = eta / (1.0 + eta * counts)
    factors = tf.expand_dims(
        tf.expand_dims(
            counts_effective_inv,
            axis=0
        ),
        axis=-1
    )
    thresholds = factors * cum_sums
    return tf.reduce_max(thresholds, axis=-2, keepdims=True)
    # mask = tf.math.greater(
    #     x=weight_inner_norm_sorted,
    #     y=thresholds
    # )
    # weight_inner_norm_filtered = tf.where(
    #     condition=mask,
    #     x=weight_inner_norm_sorted,
    #     y=tf.zeros_like(weight_inner_norm_sorted)
    # )
    # count = tf.reduce_sum(
    #     tf.cast(mask, tf.float32),
    #     axis=-2,
    #     keepdims=True
    # )
    # factor = eta / (1.0 + eta * count)
    # cum_sum = tf.reduce_sum(
    #     weight_inner_norm_filtered,
    #     axis=-2,
    #     keepdims=True
    # )
    # threshold = factor * cum_sum
    # return threshold


if __name__ == "__main__":
    graph = tf.Graph()
    eta = 1.0
    epsilon = 1e-10
    with graph.as_default():
        a = tf.reshape(
            tf.range(6, dtype=tf.float32),
            shape=[2, 3, 1]
        )
        a_new = proximal_operator_exclusive_sparse(
            weight=a,
            eta=eta,
            epsilon=epsilon
        )
        a_norm = tf.norm(
            a,
            ord='euclidean',
            axis=-1,
            keepdims=True
        )
        a_threshold = find_threshold_biased_partial_mean(
            weight_inner_norm=a_norm,
            eta=eta
        )
    sess = tf.Session(graph=graph)
    print(sess.run(a))
    print(sess.run(a_new))
    print(sess.run(a_threshold))
