import _pickle as cPickle

from preprocess import dictionary_decode


def label_feature_join(
        label_file, id_file,
        feature_file,
        label_file_out=None,
        id_file_out=None,
        feature_file_out=None
):
    with open(label_file, 'rb') as lf:
        posts = cPickle.load(lf, encoding='bytes')
        dictionary_decode(posts=posts)
    with open(id_file, "rb") as idf:
        ids = cPickle.load(idf)
        ids = list(map(lambda x: x.decode('utf-8'), ids))

    ff = open(feature_file, "r")

    if label_file_out is None:
        label_file_out = label_file + "_joined"
    lfo = open(label_file_out, 'w')
    if id_file_out is None:
        id_file_out = id_file + "_joined"
    idfo = open(id_file_out, 'w')
    if feature_file_out is None:
        feature_file_out = feature_file + "_joined"
    ffo = open(feature_file_out, 'w')

    for i in range(len(ids)):
        id = ids[i]
        print(id)
        if id not in posts:
            ff.readline()                              # without label discarded
            continue
        # valid id with label and feature #
        lfo.write(','.join(list(map(str, posts[id]["REACTIONS"]))) + "\n")
        idfo.write(id + "\n")
        ffo.write(ff.readline())

    ff.close()
    lfo.close()
    idfo.close()
    ffo.close()


if __name__ == "__main__":
    data_dir = "../data"
    label_feature_join(
        label_file=data_dir + "/posts_reactions_all",
        id_file=data_dir + "/posts_content_all_text_ids",
        feature_file=data_dir + "/posts_content_all_features_[CLS]_SUM",
    )
