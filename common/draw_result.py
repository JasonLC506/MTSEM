import numpy as np
import matplotlib.pyplot as plt


def draw_result(
        results,
        data_sizes=None,
        n_topics=1,
        topic_wise=True
):
    """
    draw results
    :param results: 1-d array of performance measure for tasks / topic-tasks
    :param data_sizes: weighted data weights, same shape with results
    :param n_topics: number of topics, 1 for non-topic case
    """
    topic_results = results.reshape([n_topics, -1])
    if not topic_wise:
        topic_results = topic_results.transpose()
    if data_sizes is not None:
        data_sizes = data_sizes.reshape([n_topics, -1])
        if not topic_wise:
            data_sizes = data_sizes.transpose()
        topic_results_weighted = topic_results * data_sizes
    else:
        topic_results_weighted = topic_results
    print("topic average: %s" % str(np.mean(topic_results_weighted, axis=1)))
    for i in range(topic_results.shape[0]):
        result = topic_results[i]
        plt.plot(result, label="topic_%d" % i if topic_wise else "task_%d" % i)
    plt.xlabel("task" if topic_wise else "topic")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    draw_result(
        np.array([1.23417833, 1.23814404, 1.23172285, 0.59684864, 1.30455169,
         1.44315786, 1.2887991, 1.22118075, 1.33352651, 1.35183431,
         1.49683191, 1.16384632, 1.29597367, 1.40639254, 1.34611033,
         1.03694362, 1.35284033, 1.51373415, 1.3603499, 1.38568453,
         1.45802832, 1.44910358, 1.76284767, 1.38260536, 1.43595775,
         1.39655122, 1.32120405, 1.49692926, 1.47951264, 1.45030631,
         1.51152303, 1.41951444, 1.52305498, 1.43438111, 1.51444433,
         1.40011191]),
        n_topics=3
    )
