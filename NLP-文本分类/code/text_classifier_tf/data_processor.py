import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from utils.utils import remove_stop_words, remove_low_words, build_vocab, build_word2vec


class DataProcessor(object):
    """数据处理类"""
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.train_x = None
        self.train_y = None
        self.eval_x = None
        self.eval_y = None
        self.test_x = None
        self.test_y = None

    def read_data(self):
        """读取数据"""
        raise NotImplementedError

    def data_process(self, inputs, labels):
        """预处理数据"""
        raise NotImplementedError

    def gen_data(self):
        """生成标准格式数据"""
        raise NotImplementedError


class O2OProcessor(DataProcessor):
    def __init__(self, train_path, test_path):
        super(O2OProcessor, self).__init__(train_path, test_path)
        self.dataset_name = 'o2o'
        self.test_id = None

        df_train, df_test = self.read_data()
        self.data_process(df_train, df_test)
        self.gen_data()

    def read_data(self):
        df_train = pd.read_csv(self.train_path, sep='\t', encoding='utf-8')
        df_train = df_train.sample(frac=1)  # shuffle
        df_test = pd.read_csv(self.test_path, sep=',', encoding='utf-8')

        return df_train, df_test

    def data_process(self, df_train, df_test):
        # 分词
        train_x = df_train['comment'].apply(lambda x: [w for w in list(jieba.cut(x)) if w != ' ']).tolist()
        train_y = df_train['label'].apply(lambda x: str(x)).tolist()
        self.test_x = df_test['comment'].apply(lambda x: [w for w in list(jieba.cut(x)) if w != ' ']).tolist()
        # 去除停用词
        all_words = remove_stop_words(train_x)
        # 切分训练集/验证集
        self.train_x, self.eval_x, self.train_y, self.eval_y = train_test_split(train_x, train_y, test_size=0.1)
        self.test_id = df_test['id'].tolist()
        # 过滤低频词
        words = remove_low_words(all_words, max_size=10000, min_freq=3)
        # 生成词汇
        word2id, label2id = build_vocab(words, self.train_y)
        # 构建词向量
        word2vec = build_word2vec(word2id)
        # 空格拼接单词
        self.train_x = list(map(lambda x: " ".join([w for w in x]), self.train_x))
        self.eval_x = list(map(lambda x: " ".join([w for w in x]), self.eval_x))
        self.test_x = list(map(lambda x: " ".join([w for w in x]), self.test_x))

    def gen_data(self):
        with open('./data/{}/train.txt'.format(self.dataset_name), 'w', encoding='utf-8') as f:
            for i in range(len(self.train_x)):
                f.write(str(self.train_x[i]) + '<SEP>' + str(self.train_y[i]) + '\n')

        with open('./data/{}/dev.txt'.format(self.dataset_name), 'w', encoding='utf-8') as f:
            for i in range(len(self.eval_x)):
                f.write(str(self.eval_x[i]) + '<SEP>' + str(self.eval_y[i]) + '\n')

        with open('./data/{}/test.txt'.format(self.dataset_name), 'w', encoding='utf-8') as f:
            for i in range(len(self.test_x)):
                f.write(str(self.test_x[i]) + '<SEP>' + str(self.test_id[i]) + '\n')


if __name__ == '__main__':
    processor = O2OProcessor('./data/o2o/train.csv', './data/o2o/test_new.csv')

