import os
from collections import Counter
import pickle
import numpy as np

# 当前预处理数据集
DATASET_NAME = "o2o"
# 停用词路径
STOPWORDS_PATH = os.path.join(os.path.dirname(os.getcwd()), 'data/stop_words.txt')
# 预处理数据保存路径
OUTPUT_PATH = os.path.join(os.path.dirname(os.getcwd()), 'data/{}'.format(DATASET_NAME))
# 预训练词向量路径
Pre_EMBEDDING_PATH = os.path.join(os.path.dirname(os.getcwd()), 'data/wv/sgns.baidubaike.bigram-char')


def remove_stop_words(inputs):
    """去除停用词"""

    if not os.path.exists(STOPWORDS_PATH):
        return [word for data in inputs for word in data if len(word) > 0]

    with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f.readlines()]
    all_words = [word for data in inputs for word in data if word not in stop_words and len(word)>0]
    return all_words


def remove_low_words(inputs, max_size=10000, min_freq=3):
    """过滤低频词"""
    word_count = Counter(inputs)  # 统计词频
    sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    sort_word_count = sort_word_count[0: max_size]
    words = [item[0] for item in sort_word_count if item[1] >= min_freq]
    return words


def build_vocab(words, labels):
    """生成词汇和标签映射表"""
    vocab = ["<PAD>", "<UNK>"] + words
    word_to_index = dict(zip(vocab, list(range(len(vocab)))))

    # 将词汇-索引映射表保存为pkl数据，之后做inference时直接加载来处理数据
    with open(os.path.join(OUTPUT_PATH, 'word2id.pkl'), 'wb') as f:
        pickle.dump(word_to_index, f)

    # 将标签-索引映射表保存为pkl数据
    unique_labels = list(set(labels))
    label_to_index = dict(zip(unique_labels, list(range(len(unique_labels)))))
    with open(os.path.join(OUTPUT_PATH, "label2id.pkl"), "wb") as f:
        pickle.dump(label_to_index, f)

    return word_to_index, label_to_index


def build_word2vec(word2id, embedding_size=300):
    """构建词向量，默认使用300维"""
    PAD = "<PAD>"
    UNK = "<UNK>"
    bound = np.sqrt(6.0) / np.sqrt(len(word2id))  # bound for random variables.

    # 默认使用随机初始化词向量
    word2vec = (1 / np.sqrt(len(word2id)) * (2 * np.random.rand(len(word2id), embedding_size) - 1))

    word2vec[word2id[PAD]] = np.zeros(embedding_size, dtype=np.float)
    word2vec[word2id[UNK]] = np.random.uniform(-bound, bound, embedding_size)

    if not os.path.exists(Pre_EMBEDDING_PATH):
        return word2vec

    # 加载预训练词向量
    with open(Pre_EMBEDDING_PATH, 'r', encoding='utf-8') as f:
        _ = f.readline()
        for line in f.readlines():
            line = line.strip()
            values = line.split(' ')
            word = values[0]
            if word in word2id.keys():
                word2vec[word2id[word]] = np.asarray([float(x) for x in values[1:]], dtype='float32')

    # 保存当前数据集词向量
    np.save(os.path.join(OUTPUT_PATH, 'word2vec.npy'), word2vec)
    return word2vec
