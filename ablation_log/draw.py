# from matplotlib import pyplot as plt

dir_names = [
    'sparsity/topic_task_sparse_layer_wise_single_layer',
    'sparsity/topic_task_sparse_layer_wise_single_layer_exclusive'
]
model_names = [
    'TMTS-el',
    'TMTS-ex'
]

# hps = [1, 2, 4, 8, 16, 32]
hps = [1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# name = '$K$'
name = '$\lambda$'
# plt.figure()
perfs_list = []
for i in range(2):
    dir_name = dir_names[i]
    perfs = []
    with open(dir_name + "/_perfs", 'r') as pf:
        for line in pf:
            perfs.append(
                100 * (1 - float(line.rstrip().split("\t")[1].split("|")[0]))
            )
    perfs_list.append(perfs)
    # plt.plot(hps, perfs, label=model_names[i])
for line_list in zip(hps, *perfs_list):
    print("\t".join(list(map(str, line_list))))
# plt.xscale('log')
# plt.xlabel(name)
# plt.ylabel('miss-classification rate')
# plt.legend()
# plt.show()
