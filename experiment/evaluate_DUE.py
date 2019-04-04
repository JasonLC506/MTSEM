import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


def evaluate(preds, trues):
    N_labels = preds.shape[-1]
    N_samps = preds.shape[0]
    print("N_samps: %d, N_labels: %d" % (N_samps, N_labels))

    # log_ppl or - log likelihood#
    log_ppl = - np.sum(np.log(preds[np.arange(N_samps), trues])) / N_samps

    # accuracy #
    accuracy = np.sum(np.argmax(preds, axis=-1) == trues) * 1.0 / N_samps

    # AUC-PR #
    trues_onehot = np.identity(N_labels)[trues,:]
    auc_pr_micro = average_precision_score(y_true=trues_onehot, y_score=preds, average='micro')
    auc_pr_macro = average_precision_score(y_true=trues_onehot, y_score=preds, average='macro')
    # auc_pr_micro = roc_auc_score(y_true=trues_onehot, y_score=preds, average='micro')
    # auc_pr_macro = roc_auc_score(y_true=trues_onehot, y_score=preds, average='macro')

    return [-log_ppl, accuracy, auc_pr_micro, auc_pr_macro]
