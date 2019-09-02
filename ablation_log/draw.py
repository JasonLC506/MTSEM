from matplotlib import pyplot as plt

dir_names = [
    'topic_dim/topic_task_sparse_layer_wise_single_layer',
    'topic_dim/topic_task_sparse_layer_wise_single_layer_exclusive'
]
model_names = [
    'TMTS-el',
    'TMTS-ex'
]

hps = [1, 2, 4, 8, 16, 32]
name = '$K$'
plt.figure()
for i in range(2):
    dir_name = dir_names[i]
    perfs = []
    with open(dir_name + "/_perfs", 'r') as pf:
        for line in pf:
            perfs.append(
                1 - float(line.rstrip().split("\t")[1].split("|")[0])
            )
    plt.plot(hps, perfs, label=model_names[i])
# plt.xscale('log')
plt.xlabel(name)
plt.ylabel('miss-classification rate')
plt.legend()
plt.show()
