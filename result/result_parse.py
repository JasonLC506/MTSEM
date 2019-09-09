import os
import numpy as np

#
# from models import Models
Models = [
    'separate',
    'shared_bottom',
    'fc',
    'inter_task_l2',
    'dmtrl_Tucker',
    'multilinear_relationship_network',
    'cross_stitch',
    'mmoe',
    'topic_task_sparse_layer_wise',
    'topic_task_sparse_layer_wise_single_layer',
    'topic_task_sparse_layer_wise_exclusive',
    'topic_task_sparse_layer_wise_single_layer_exclusive'
]
# for model in Model_names:
#     if model not in Models:
#         print('%s not in' % model)

Model_names = [
    'Separate',
    'Shared-bottom',
    'Single',
    'Inter-task-l2',
    'DMTRL',
    'MRN',
    'Cross-stitch',
    'MMOE',
    'TMTS-el',
    'TMTS-el-1',
    'TMTS-ex',
    'TMTS-ex-1'
]


def parse(data_name):
    results = {}
    stds = {}
    dir_name = '../result/' + data_name
    for rf_name in os.listdir(dir_name):
        if rf_name not in Models:
            print("file '%s' not a model result file" % rf_name)
            continue
        with open(dir_name + "/" + rf_name, 'r') as rf:
            for line in rf:
                if 'perf' in line:
                    print(line.rstrip() + '--perf line')
                    perf = parse_line(line)
                    results[rf_name] = perf
                elif 'std' in line:
                    print(line.rstrip() + '--std line')
                    std = parse_line(line)
                    stds[rf_name] = std
    results_list = []
    stds_list = []
    for model in Models:
        results_list.append(results[model])
        stds_list.append(stds[model])
    return np.array(results_list), np.array(stds_list)


def parse_line(line):
    perf = line.rstrip().split(':')[1].split(',')
    perf = np.array(list(map(
        float,
        perf
    )))
    return perf


def list_final(data_names, spliter='& ', line_spliter='\\\\\n', to_misclass=True):
    results_list = []
    stds_list = []
    for data_name in data_names:
        results, stds = parse(data_name)
        results_list.append(results[:, -1])
        stds_list.append(stds[:, -1])
    print(spliter.join(['data'] + data_names) + line_spliter)
    for i in range(len(Models)):
        results = [results_list[d][i] for d in range(len(data_names))]
        if to_misclass:
            results = list(map(lambda x: "%4.2f" % ((1.0 - x) * 100.0), results))
        else:
            results = list(map(lambda x: "%6.3f" % x, results))
        print(spliter.join([Model_names[i]] + results) + line_spliter)



def list_per_task_comparison(data_names, spliter='& ', line_spliter='\\\\\n', to_misclass=True):
    results_list = []
    stds_list = []
    for data_name in data_names:
        results, stds = parse(data_name)

        separate_id = Models.index('separate')
        result_standard = results[separate_id]
        if to_misclass:
            results_normalized = (results - result_standard)/np.expand_dims(result_standard, axis=0)
        else:
            results_normalized = - (results - result_standard)/np.expand_dims(result_standard, axis=0)

        # with open(data_name + "_per_task_result", 'w') as f:
        #     lines = []
        #     for t in range(results_normalized.shape[1]):
        #         lines.append('\t'.join([str(t)] + list(map(lambda x: '%f' % (100 * x), results_normalized[:,t]))))
        #     f.write('\n'.join(lines))
        results_list.append(np.mean(results_normalized, axis=-1))
    print(spliter.join(['data'] + data_names) + line_spliter)
    for i in range(len(Models)):
        results = [results_list[d][i] for d in range(len(data_names))]
        results = list(map(lambda x: "%4.2f" % (x * 100), results))
        print(spliter.join([Model_names[i]] + results) + line_spliter)


            # comparison_list = []
    # for i in range(len(data_names)):
    #     results = results_list[i]
    #     separate_id = Models.index('separate')
    #     result_standard = results[separate_id]
    #     if to_misclass:
    #         results_compare = np.sum((results <= result_standard).astype(np.int64), axis=-1)
    #     else:
    #         results_compare = np.sum((results >= result_standard).astype(np.int64), axis=-1)
    #
    #     comparison_list.append(results_compare)
    # results_list = comparison_list
    # print(spliter.join(['data'] + data_names) + line_spliter)
    # for i in range(len(Models)):
    #     results = [results_list[d][i] for d in range(len(data_names))]
    #     results = list(map(lambda x: "%d" % x, results))
    #     print(spliter.join([Models[i]] + results) + line_spliter)


if __name__ == "__main__":
    list_final(
        data_names=['MNIST_MTL', 'synthetic_topic_task_sparse_v2', 'AwA2', 'school'],
        # data_names = ['AwA2'],
        to_misclass=True
    )
    # list_per_task_comparison(
    #     # data_names=['MNIST_MTL', 'synthetic_topic_task_sparse_v2']
    #     data_names=['SEM'], to_misclass=False
    # )


