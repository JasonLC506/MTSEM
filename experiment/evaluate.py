import numpy as np


def simple_evaluate(
        model,
        data,
        per_task=False,
):
    results = model.test(
        data_generator=data
    )
    perf = performance(
        model_local=model,
        data_local=data,
        per_task=per_task
    )
    return {
        "perf": perf,
        "results": results
    }


def performance(
        model_local,
        data_local,
        per_task=False
):
    preds = model_local.predict(
        data_generator=data_local
    )
    labels = []
    tasks = []
    for data_batched in data_local.generate(
            batch_size=model_local.model_spec["batch_size"],
            random_shuffle=False
    ):
        labels.append(data_batched["label"])
        tasks.append(data_batched["task"])
    labels = np.concatenate(labels, axis=0)
    tasks = np.concatenate(tasks, axis=0)

    # one-hot to index #
    trues = labels

    perf = accuracy(
        preds=preds,
        trues=trues,
        tasks=None if not per_task else tasks
    )
    return perf


def accuracy(
        preds,
        trues,
        tasks=None
):
    """
    all in one-hot prob distribution
    :param preds:
    :param trues:
    :param tasks: one-hot
    :return:
    """
    preds_label = np.argmax(preds, axis=-1).squeeze()
    trues_label = np.argmax(trues, axis=-1).squeeze()
    acc = np.sum(preds_label == trues_label) / float(np.sum(np.ones_like(preds_label)))
    if tasks is None:
        return acc
    else:
        accs = []
        accs.append(np.sum(tasks[np.where(preds_label == trues_label)[0], :], axis=0) / np.sum(tasks, axis=0))
        accs.append(np.array([acc]))
        return np.concatenate(accs, axis=0)
