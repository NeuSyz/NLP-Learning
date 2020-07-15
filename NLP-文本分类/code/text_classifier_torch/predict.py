import pickle
import os
import sys
import numpy as np
from tkinter import _flatten
import argparse
import json
from models import TextCnnModel, BiLstmModel, BiLstmAttenModel, RCnnModel, Transformer
import torch
import torch.nn as nn
sys.path.append(os.path.abspath(os.getcwd()))


class Predictor(object):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_path = os.path.join(os.path.abspath(os.getcwd()),
                                        config["data_path"])

        self.word_to_index, self.label_to_index = self.load_vocab()
        self.index_to_label = {value: key for key, value in self.label_to_index.items()}
        self.vocab_size = len(self.word_to_index)
        self.word_vectors = None
        self.sequence_length = self.config["sequence_length"]

        # 加载预训练词向量
        self.load_embedding()
        # 创建模型
        self.create_model()
        # 加载计算图
        self.load_model()

    def load_embedding(self):
        if os.path.exists(os.path.join(self.output_path, "word2vec.npy")):
            print("load word_vectors")
            embedding = np.load(os.path.join(self.output_path, "word2vec.npy"))
            self.word_vectors = torch.tensor(embedding.astype('float32'))

    def load_vocab(self):
        # 将词汇-索引映射表加载出来
        with open(os.path.join(self.output_path, "word2id.pkl"), "rb") as f:
            word_to_index = pickle.load(f)

        with open(os.path.join(self.output_path, "label2id.pkl"), "rb") as f:
            label_to_index = pickle.load(f)

        return word_to_index, label_to_index

    def sentence_to_idx(self, sentence):
        """
        将分词后的句子转换成idx表示
        :param sentence:
        :return:
        """
        sentence_ids = [self.word_to_index.get(token, self.word_to_index["<UNK>"]) for token in sentence]
        sentence_pad = sentence_ids[: self.sequence_length] if len(sentence_ids) > self.sequence_length \
            else sentence_ids + [0] * (self.sequence_length - len(sentence_ids))
        sentence_len = len(sentence) if len(sentence) < self.sequence_length else self.sequence_length
        return sentence_pad, sentence_len

    def init_network(self, method='xavier', exclude='embedding'):
        """
        权重初始化，默认xavier
        :param method:
        :param exclude:
        :return:
        """
        for name, w in self.model.named_parameters():
            if exclude not in name:
                if 'weight' in name:
                    if method == 'xavier':  # 输入和输出方差相同，包括前向传播和后向传播
                        nn.init.xavier_normal_(w)
                    elif method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        if self.config["model_name"] == "textcnn":
            self.model = TextCnnModel(config=self.config, vocab_size=self.vocab_size,
                                      word_vectors=self.word_vectors).to(self.device)
            self.init_network()
        elif self.config["model_name"] == "bilstm":
            self.model = BiLstmModel(config=self.config, vocab_size=self.vocab_size,
                                     word_vectors=self.word_vectors).to(self.device)
            self.init_network()
        elif self.config["model_name"] == "bilstm_atten":
            self.model = BiLstmAttenModel(config=self.config, vocab_size=self.vocab_size,
                                          word_vectors=self.word_vectors).to(self.device)
            self.init_network()
        elif self.config["model_name"] == "rcnn":
            self.model = RCnnModel(config=self.config, vocab_size=self.vocab_size,
                                   word_vectors=self.word_vectors).to(self.device)
            self.init_network()
        elif self.config["model_name"] == "transformer":
            self.model = Transformer(config=self.config, vocab_size=self.vocab_size,
                                     word_vectors=self.word_vectors).to(self.device)

    def load_model(self):
        """
        加载训练模型
        :return:
        """
        save_path = os.path.join(os.path.abspath(os.getcwd()),
                                 self.config["ckpt_model_path"])
        model_save_path = os.path.join(save_path, self.config["model_name"] + ".pkl")
        self.model.load_state_dict(torch.load(model_save_path))

    def predict(self, sentences):
        """
        给定句子，预测分类结果
        :param sentences:
        :return:
        """
        sentences_ids = [self.sentence_to_idx(sent)[0] for sent in sentences]
        sentences_lens = [self.sentence_to_idx(sent)[1] for sent in sentences]
        x = np.array(sentences_ids, dtype='int64')
        x_len = np.array(sentences_lens, dtype='int64')

        x = torch.from_numpy(x).long().to(self.device)
        x_len = torch.from_numpy(x_len).long().to(self.device)
        batch_x = (x, x_len)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch_x)
            pred = torch.max(outputs.data, 1)[1].cpu().numpy()
        predictions = pred.tolist()
        predictions = _flatten(predictions)
        labels = [self.index_to_label[pred] for pred in predictions]

        return labels


if __name__ == '__main__':
    data = [['一如既往', '的', '不好', '，', '菜品', '不好', '，', '服务', '不好', '！']]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config/textcnn.json", help="config path of model")
    args = parser.parse_args()
    with open(os.path.join(os.getcwd(), args.config_path), "r") as fr:
        config = json.load(fr)
    predictor = Predictor(config)
    res = predictor.predict(data)
    print(res)