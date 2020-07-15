import os
import pickle
import numpy as np


class DataHelper(object):
    """数据管理类"""
    def __init__(self, config, train=True):
        self.config = config
        self.train = train
        self._train_data_path = os.path.join(os.getcwd(), '{}/train.txt'.format(config["data_path"]))
        self._eval_data_path = os.path.join(os.getcwd(), '{}/dev.txt'.format(config["data_path"]))
        self._data_save_path = os.path.join(os.getcwd(), '{}'.format(config["data_path"]))
        self.word2vec = None
        self.word2id = None
        self.label2id = None

    def read_data(self):
        """
        读取数据
        :return: 返回分词后的文本内容和标签，inputs = [[]], labels = []
        """
        inputs = []
        labels = []
        data_path = self._train_data_path if self.train else self._eval_data_path

        with open(data_path, "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                try:
                    text, label = line.strip().split("<SEP>")
                    inputs.append(text.strip().split(" "))
                    labels.append(label)
                except:
                    print(10*"*" + "读取有错误" + 10*"*")
                    continue

        return inputs, labels

    def trans_to_index(self, inputs):
        """
        将输入转化为索引表示
        :param inputs: 输入
        :return:
        """
        inputs_idx = [[self.word2id.get(word, self.word2id["<UNK>"]) for word in sentence] for sentence in inputs]

        return inputs_idx

    def trans_label_to_index(self, labels):
        """
        将标签也转换成数字表示
        :param labels: 标签
        :return:
        """
        labels_idx = [self.label2id[label] for label in labels]
        return labels_idx

    def padding(self, inputs, sequence_length):
        """
        对序列进行截断和补全
        :param inputs: 输入
        :param sequence_length: 预定义的序列长度
        :return:
        """
        new_inputs = [sentence[:sequence_length]
                      if len(sentence) > sequence_length
                      else sentence + [0] * (sequence_length - len(sentence))
                      for sentence in inputs]

        return new_inputs

    def gen_data(self):
        """
        生成可导入到模型中的数据
        :return:
        """
        # 1，读取原始数据
        inputs, labels = self.read_data()
        print("read finished")
        # 2，读取预处理文件
        with open(os.path.join(self._data_save_path, "word2id.pkl"), "rb") as f:
            self.word2id = pickle.load(f)
        with open(os.path.join(self._data_save_path, "label2id.pkl"), "rb") as f:
            self.label2id = pickle.load(f)

        self.word2vec = np.load(os.path.join(self._data_save_path, "word2vec.npy")) if self.train else None
        print("vocab,embedding process finished")

        # 3，输入转索引
        inputs_idx = self.trans_to_index(inputs)
        print("index transform finished")

        # 4，对输入做padding
        inputs_idx = self.padding(inputs_idx, self.config['sequence_length'])
        print("padding finished")

        # 5，标签转索引
        labels_idx = self.trans_label_to_index(labels)
        print("label index transform finished")

        return np.array(inputs_idx), np.array(labels_idx)

    def next_batch(self, x, y, batch_size):
        """
        生成batch数据集
        :param x: 输入
        :param y: 标签
        :param batch_size: 批量的大小
        :return:
        """
        perm = np.arange(len(x))
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]

        num_batches = len(x) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = np.array(x[start: end], dtype="int64")
            batch_y = np.array(y[start: end], dtype="float32")

            yield dict(x=batch_x, y=batch_y)