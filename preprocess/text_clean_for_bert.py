import _pickle as cPickle


def dictionary_decode(posts):
    for d_id in posts:
        post_new = {}
        for key in posts[d_id]:
            value = posts[d_id][key]
            if isinstance(value, str):
                post_new[key.decode('utf-8')] = value.decode('utf-8')
            else:
                post_new[key.decode('utf-8')] = value
        posts[d_id] = post_new


def posts_content_clean(posts):
    """
    clean post content into list of documents written to file split by '\n'
    :param posts: dictionary of posts {id: {"MESSAGE": string, "url": string}}
    :return: documents: list of strings (without any "\n" within a document); post_ids: list of post_ids
    """
    post_ids = []
    documents = []
    for d_id in posts:
        post_ids.append(d_id)
        document = posts[d_id]["MESSAGE"].replace("\n", " ")
        # print(document) ### test ###
        documents.append(document)
    return documents, post_ids


if __name__ == "__main__":
    input_file = "../data/posts_content_all"
    output_file = "../data/posts_content_all_text"
    output_file_post_id = "../data/posts_content_all_text_ids"

    posts = cPickle.load(open(input_file, 'rb'), encoding="bytes")
    dictionary_decode(posts)
    documents, post_ids = posts_content_clean(posts)
    with open(output_file, "wb") as of:
        for doc in documents:
            print(doc) ### test ###
            of.write((doc + "\n").encode('ascii', 'ignore'))
    with open(output_file_post_id, "wb") as of_pid:
        cPickle.dump(post_ids, of_pid)
