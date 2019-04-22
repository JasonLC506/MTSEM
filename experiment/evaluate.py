import numpy as np


def simple_evaluate(
        model,
        data
):
    results = model.test(
        data_generator=data
    )
    perf = performance(
        model_local=model,
        data_local=data
    )
    return {
        "perf": perf,
        "results": results
    }


def performance(
        model_local,
        data_local
):
    preds = model_local.predict(
        data_generator=data_local
    )
    labels = []
    for data_batched in data_local.generate(
            batch_size=model_local.model_spec["batch_size"],
            random_shuffle=False
    ):
        labels.append(data_batched["label"])
    labels = np.concatenate(labels, axis=0)
    # one-hot to index #
    trues = labels

    perf = accuracy(
        preds=preds,
        trues=trues
    )
    return perf


def accuracy(
        preds,
        trues
):
    """
    all in one-hot prob distribution
    :param preds:
    :param trues:
    :return:
    """
    preds_label = np.argmax(preds, axis=-1).squeeze()
    trues_label = np.argmax(trues, axis=-1).squeeze()
    acc = np.sum(preds_label == trues_label) / float(np.sum(np.ones_like(preds_label)))
    return acc
