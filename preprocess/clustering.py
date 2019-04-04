import json
import numpy as np
import lda
import _pickle as cPickle
from nltk.corpus import stopwords


STOPWORDS = set(stopwords.words("english"))


def tokenization_extract_bert(feature_file, output_file, cls_seq_remove=True):
    """
    extract tokenization from BERT output feature file
    """
    ff = open(feature_file, "r")
    of = open(output_file, "w")
    for line in ff:
        info = json.loads(line.rstrip())
        tokens = list(map(lambda x: x["token"], info["features"]))
        if cls_seq_remove:
            tokens = tokens[1: -1]
        of.write("\t".join(tokens) + "\n")
    ff.close()
    of.close()


def doc_index(doc_file, freq_low_threshold=5, freq_up_threshold=1.0):
    doc_list = []
    with open(doc_file, "r") as df:
        for line in df:
            doc = line.rstrip().split("\t")
            doc_list.append(doc)

    # word_dict #
    word_dict = {}                          # [index, count]
    cnt = 0
    for doc in doc_list:
        for word in doc:
            if word in word_dict:
                word_dict[word][1] += 1
            else:
                word_dict[word] = [cnt, 1]
                cnt += 1
    print("total words: %d, tokens: %d, documents: %d" % (cnt, sum(list(map(len, doc_list))), len(doc_list)))
    freq_up_threshold = int(freq_up_threshold * len(doc_list))
    for word in word_dict:
        if word_dict[word][1] < freq_low_threshold or \
                        word_dict[word][1] > freq_up_threshold or \
                        word in STOPWORDS or \
                        "#" in word:
            word_dict[word][0] = -1                       # paddle index
    cnt = 0
    for word in word_dict:
        if word_dict[word][0] != -1:
            word_dict[word][0] = cnt
            cnt += 1
    print("total words: %d, tokens: %d, documents: %d" % (cnt, sum(list(map(len, doc_list))), len(doc_list)))

    # index doc #
    doc_indexed_list = []
    for doc in doc_list:
        doc_indexed = map(lambda x: word_dict[x][0], doc)
        doc_indexed_list.append(doc_indexed)

    # matrix #
    doc_matrix = np.zeros([len(doc_list), cnt + 1], dtype=np.float32)
    for i in range(len(doc_list)):
        doc_indexed = doc_indexed_list[i]
        for w_index in doc_indexed:
            doc_matrix[i][w_index] += 1

    return doc_matrix[:, :-1], doc_indexed_list, word_dict


def doc_lda(
        doc_matrix,
        n_topics=5,
        n_iter=1000,
        vocab=None,
        doc_topic_file=None
):
    model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
    model.fit(doc_matrix)
    if vocab:
        topic_word = model.topic_word_
        n_top_words = 20
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
            print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    if doc_topic_file:
        with open(doc_topic_file, "wb") as dtf:
            cPickle.dump(model.doc_topic_, dtf)


def word_dict2vocab(word_dict):
    vocab = ["NoneWord" for _ in range(len(word_dict))]
    for word in word_dict:
        word_index = word_dict[word]
        if isinstance(word_index, list):
            word_index = word_index[0]
        vocab[word_index] = word
    return vocab


def merge_doc_topic_ids(doc_topic_file, id_file, doc_topic_dict_file=None):
    doc_topic = cPickle.load(open(doc_topic_file, 'rb'))
    ids = cPickle.load(open(id_file, 'rb'))
    ids = list(map(lambda x: x.decode('utf-8'), ids))
    doc_topic_dict = dict()
    assert doc_topic.shape[0] == len(ids)
    for i in range(len(ids)):
        doc_topic_dict[ids[i]] = doc_topic[i]
    if doc_topic_dict_file is None:
        doc_topic_dict_file = doc_topic_file + "_withids"
    with open(doc_topic_dict_file, 'wb') as dtdf:
        cPickle.dump(doc_topic_dict, dtdf)


if __name__ == "__main__":
    # tokenization_extract_bert(
    #     feature_file="../data/features_posts_content_all.json",
    #     output_file="../data/posts_content_tokenized_bert"
    # )

    # doc_matrix, doc_indexed_list, word_dict = doc_index(
    #     doc_file="../data/posts_content_tokenized_bert",
    #     freq_low_threshold=5,
    #     freq_up_threshold=0.5
    # )
    # print("word_cnt histogram: ")
    # print(np.histogram(list(map(lambda x: x[1], word_dict.values()))))
    #
    # vocab = word_dict2vocab(word_dict)
    # doc_lda(
    #     doc_matrix=doc_matrix.astype(dtype=np.int64),
    #     n_topics=3,
    #     vocab=vocab,
    #     doc_topic_file="../data/posts_content_topics_pkl"
    # )

    merge_doc_topic_ids(
        doc_topic_file="../data/posts_content_topics_pkl",
        id_file="../data/posts_content_all_text_ids"
    )
