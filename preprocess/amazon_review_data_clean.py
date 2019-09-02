import os
import numpy as np


def extract_user_data(user_file):
    user_data = []
    with open(user_file) as f:
        f.readline()  # user name
        cnt = 0
        review = {}
        for line_ in f:
            cnt += 1
            line = line_.rstrip()
            if cnt % 5 == 1:
                review["product_id"] = line
            elif cnt % 5 == 2:
                review["content"] = line
            elif cnt % 5 == 3:
                review["category"] = line
            elif cnt % 5 == 4:
                review["label"] = line
            else:
                # no need of post time #
                user_data.append(review)
                review = {}
    return user_data


def extract_users_data(users_dir):
    users_data = {}
    for fn in os.listdir(users_dir):
        user_id = fn[:-4]
        filename = os.path.join(users_dir, fn)
        users_data[user_id] = extract_user_data(user_file=filename)
    print("n_users: %d, n_reviews: %d" % (len(users_data), sum(list(map(len, users_data.values())))))
    return users_data


def count_task(
        users_data,
        task_identifier="category"
):
    task_identifier_index = ["user_id", "product_id", "category"].index(task_identifier)
    task_id_count = dict()
    for user_id in users_data:
        for user_data in users_data[user_id]:
            identifier_values = [user_id, user_data['product_id'], user_data['category']]
            task_id = identifier_values[task_identifier_index]
            if task_id == "":
                continue  # meaningful task_id only
            task_id_count.setdefault(task_id, 0)
            task_id_count[task_id] += 1
    print("done with # tasks: %d" % len(task_id_count))
    for task_id in task_id_count:
        print("task: '%s': %d" % (task_id, task_id_count[task_id]))
    return task_id_count


def save_as_data(
        users_data,
        target_dir,
        task_identifier="category",
        task_id_count={},
        task_id_count_threshold=1000
):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    task_identifier_index = ["user_id", "product_id", "category"].index(task_identifier)
    task_id_count_filtered = dict()
    fn_s = ["content", "label", "id"]
    fns = list(map(lambda x: os.path.join(target_dir, x), fn_s))
    fs = list(map(lambda x: open(x, 'wb'), fns))
    for user_id in users_data:
        for user_data in users_data[user_id]:
            identifier_values = [user_id, user_data['product_id'], user_data['category']]
            task_id = identifier_values[task_identifier_index]
            if task_id not in task_id_count:
                continue
            if task_id_count[task_id] < task_id_count_threshold:
                continue
            task_id_count_filtered.setdefault(task_id, 0)
            task_id_count_filtered[task_id] += 1
            rest_id = "_".join(
                identifier_values[:task_identifier_index] + identifier_values[task_identifier_index + 1:]
            )
            object_id = task_id + "_" + rest_id
            content = user_data["content"]
            label = user_data["label"]
            data_to_write = [content, label, object_id]
            for i in range(len(data_to_write)):
                fs[i].write((data_to_write[i] + "\n").encode('ascii', 'ignore'))
    for f in fs:
        f.close()
    print("done with # tasks: %d" % len(task_id_count_filtered))
    for task_id in task_id_count_filtered:
        print("task: '%s': %d" % (task_id, task_id_count[task_id]))


def label_transform(label_file):
    """
    from int 1-5 scale to 2 classes
    """
    dist_count = dict()
    with open(label_file, 'r') as f:
        with open(label_file + "_transformed", 'w') as tf:
            for line in f:
                score = int(line.rstrip())
                dist_count.setdefault(score, 0)
                dist_count[score] += 1
                if score <= 3:
                    label = [1.0, 0.0]
                else:
                    label = [0.0, 1.0]
                tf.write(','.join(list(map(str, label))) + "\n")
    print(dist_count)


if __name__ == "__main__":
    # users_data = extract_users_data("../data/Amazon_review/Users")
    # task_id_count = count_task(
    #     users_data=users_data,
    #     task_identifier='user_id'
    # )
    # print(np.histogram(list(task_id_count.values())))
    # save_as_data(
    #     users_data=users_data,
    #     target_dir="../data/Amazon_review/raw_user_task",
    #     task_identifier='user_id',
    #     task_id_count=task_id_count,
    #     task_id_count_threshold=70
    # )

    label_transform(label_file='../data/Amazon_review/raw_user_task/label')
