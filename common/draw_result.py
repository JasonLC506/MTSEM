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


from mpl_toolkits.mplot3d import Axes3D


def draw_3d_bar(data, x_name="task", y_name="topic", z_name="norm", name="barchat"):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    data = np.transpose(data)
    _x = np.arange(data.shape[0])
    _y = np.arange(data.shape[1])
    _xx, _yy = np.meshgrid(_x, _y)
    x0, y0 = _xx.ravel(), _yy.ravel()
    print(x0)
    print(y0)
    top = data[x0, y0]
    print(top)
    # nnz_index = np.argwhere(top > 1e-10).squeeze(axis=-1)
    # print(nnz_index)
    # x = x0[nnz_index]
    # y = y0[nnz_index]
    x = x0
    y = y0
    print(x)
    print(y)
    top_origin = data[x, y]
    print(top)
    top_max_origin = np.max(top_origin)
    top_log = np.log(top_origin*1e+10)
    top_max_log = np.max(top_log)
    top = top_log / top_max_log
    bottom = np.zeros_like(top)
    width = depth = 0.5
    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax1.set_xlabel(x_name)
    ax1.set_xticklabels(["%d" % i for i in _x.tolist()])
    ax1.set_xticks(_x.tolist())
    ax1.set_ylabel(y_name)
    ax1.set_yticklabels(["%d" % j for j in _y.tolist()])
    ax1.set_yticks(_y.tolist())
    ax1.set_zlabel(z_name)
    ax1.set_zticks([0.0, 1.0])
    ax1.set_zticklabels([0.0, "%1.0e" % top_max_origin])
    ax1.set_title(name)
    ax1.view_init(elev=67, azim=-67)
    plt.show()


if __name__ == "__main__":
    draw_3d_bar(
        data=np.array(
            [[0., 0.00018888, 0.00144995, 0.00012558, 0.02114133,
              0., 0.02929572, 0., 0.00013544, 0.00123719],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.],
             [0., 0., 0., 0.00019292, 0.,
              0.00141796, 0., 0., 0., 0.00044056],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.],
             [0.05279082, 0., 0.00125791, 0.00012318, 0.,
              0.00169347, 0., 0., 0., 0.0005632],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.],
             [0., 0., 0.00082174, 0., 0.,
              0.00011927, 0., 0.02673253, 0., 0.00232163],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.],
             [0.00094449, 0.0085556, 0.02249996, 0.04431492, 0.0091062,
              0.01411574, 0.02198456, 0.01011672, 0.03997799, 0.03958007]]
        ),
        name=r'$\lambda=1e-2$'
    )
    # draw_result(
    #     np.array([1.23417833, 1.23814404, 1.23172285, 0.59684864, 1.30455169,
    #      1.44315786, 1.2887991, 1.22118075, 1.33352651, 1.35183431,
    #      1.49683191, 1.16384632, 1.29597367, 1.40639254, 1.34611033,
    #      1.03694362, 1.35284033, 1.51373415, 1.3603499, 1.38568453,
    #      1.45802832, 1.44910358, 1.76284767, 1.38260536, 1.43595775,
    #      1.39655122, 1.32120405, 1.49692926, 1.47951264, 1.45030631,
    #      1.51152303, 1.41951444, 1.52305498, 1.43438111, 1.51444433,
    #      1.40011191]),
    #     n_topics=3
    # )
