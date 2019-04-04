import json
import numpy as np
from datetime import datetime


def feature_extract_doc(doc_features, tokens="[CLS]", aggregation="SUM"):
    """
    :param doc_features: {
                            "linex_index": int,
                            "features":
                               [{"token": string, "layers: ["index": int, "values": [float]]}]
                         }
    :param tokens:
    :param aggregation:
    :return:
    """
    # token extract #
    if tokens == "[CLS]":
        feature_layers = list(map(lambda x: x["values"], doc_features["features"][0]["layers"]))
        feature_layers = np.array(feature_layers, np.float32)
    else:
        raise NotImplementedError

    # aggregation #
    if aggregation == "SUM":
        feature = np.sum(feature_layers, axis=0)
    else:
        raise NotImplementedError

    return feature


def feature_extract(feature_file, output_file, tokens="[CLS]", aggregation="SUM"):
    """
    extract features for each document in feature output file from BERT
    :param feature_file: path of feature file
    :param output_file: path of output file
    :param tokens: one of ["CLS", "ALL", "ALL_CONCATE", "ALL_SUM"] as which tokens to use sequence-wise
    :param aggregation: aggregation method, one of ["SUM", "CONCATE"] for different layers
    :return: output_file as line-by-line written for memory
    """
    ff = open(feature_file, 'r')
    of = open(output_file + "_" + tokens + "_" + aggregation, 'w')
    cnt = 0
    start = datetime.now()
    for line in ff:
        info = json.loads(line.rstrip())
        # dict_deep_print(info)
        # break
        feature = feature_extract_doc(info, tokens=tokens, aggregation=aggregation)
        if np.ndim(feature) != 1:
            raise ValueError("feature dimension must be 1 instead of %d" % np.ndim(feature))
        of.write(",".join(map(str, feature.tolist())) + "\n")
        cnt += 1
        if cnt % 1000 == 0:
            end = datetime.now()
            print("%d takes %f seconds" % (cnt, (end - start).total_seconds()))
    of.close()
    ff.close()


def dict_deep_print(dictionary, key_only=True, indent=""):
    if not key_only:
        raise NotImplementedError
    for k in dictionary:
        print(indent + str(k))
        if isinstance(dictionary[k], dict):
            dict_deep_print(dictionary[k], key_only=key_only, indent=indent + "\t")
        elif isinstance(dictionary[k], list):
            print(indent + "\t" + "length: %d" % len(dictionary[k]))
        else:
            print(indent + "\t" + "type: %s" % str(type(dictionary[k])))


def test_output(output_file):
    with open(output_file) as of:
        cnt = 0
        old_feature_dim = None
        for line in of:
            feature = line.rstrip().split(",")
            cnt += 1
            if cnt == 1:
                print("feature dimension: %d" % len(feature))
                old_feature_dim = len(feature)
            if old_feature_dim is not None:
                assert old_feature_dim == len(feature)
        print("total instances: %d" % cnt)


if __name__ == "__main__":
    # feature_extract(
    #     feature_file="../bert-master/output/features_posts_content_all.json",
    #     output_file="../data/posts_content_all_features",
    #     tokens="[CLS]",
    #     aggregation="SUM"
    # )
    test_output(output_file="../data/posts_content_all_features" + "_[CLS]" + "_SUM")
