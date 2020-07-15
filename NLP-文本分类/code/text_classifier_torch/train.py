import json
import os
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
from utils.data_helper import DataHelper
from models import TextCnnModel, BiLstmModel, BiLstmAttenModel, RCnnModel, Transformer
sys.path.append(os.path.abspath(os.getcwd()))


class Trainer(object):

    def __init__(self, args):
        self.args = args
        with open(os.path.join(os.path.abspath(os.getcwd()), args.config_path), "r") as fr:
            self.config = json.load(fr)

        self.train_helper = None
        self.eval_helper = None
        self.word_vectors = None
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载数据集
        self.load_data()
        self.train_inputs, self.train_labels, self.train_inputs_len = self.train_helper.gen_data()
        print("train data size: {}".format(len(self.train_labels)))
        self.vocab_size = len(self.train_helper.word2id)
        print("vocab size: {}".format(self.vocab_size))
        self.label_list = [value for key, value in self.train_helper.label2id.items()]
        # 初始化模型预训练词向量
        self.init_pretrained_embedding(self.train_helper.word2vec)  # 加载预训练词向量

        self.eval_inputs, self.eval_labels, self.eval_inputs_len = self.eval_helper.gen_data()
        print("eval data size: {}".format(len(self.eval_labels)))
        print("label numbers: ", len(self.label_list))
        # 初始化模型对象
        self.create_model()

    def load_data(self):
        """
        创建数据对象
        :return:
        """
        # 生成训练集对象并生成训练数据
        self.train_helper = DataHelper(self.config, self.device, train=True)

        # 生成验证集对象和验证集数据
        self.eval_helper = DataHelper(self.config, self.device, train=False)

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

    def init_pretrained_embedding(self, embedding):
        """
        初始化预训练词向量
        :param embedding:
        :return:
        """
        self.word_vectors = torch.tensor(embedding.astype('float32'))

    def get_optimizer(self):
        """
        定义优化器
        :return:
        """
        optimizer = None
        if self.config["optimization"] == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        if self.config["optimization"] == "rmsprop":
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config["learning_rate"])
        if self.config["optimization"] == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["learning_rate"])
        return optimizer

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

    def train(self):
        """训练"""
        self.model.train()
        self.optimizer = self.get_optimizer()

        total_batch = 0
        for epoch in range(self.config["epochs"]):
            print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))
            for batch_x, batch_y in self.train_helper.next_batch(self.train_inputs, self.train_labels,
                                                                 self.config["batch_size"], self.train_inputs_len):
                total_batch += 1
                outputs = self.model(batch_x)
                self.model.zero_grad()
                loss = F.cross_entropy(outputs, batch_y)
                if total_batch % 5 == 0:
                    # true = batch_y.data.cpu()
                    # pred = torch.max(outputs.data, 1)[1].cpu()
                    # train_acc = metrics.accuracy_score(true, pred)
                    print("Train: step:{}, loss:{}".format(total_batch, loss))
                loss.backward()
                self.optimizer.step()

                if self.eval_helper and total_batch % self.config["eval_every"] == 0:

                    dev_acc, dev_loss = self.evaluate()
                    print("\n")
                    print("Eval: loss:{}, acc:{}".format(dev_loss, dev_acc))
                    print("\n")
                    if self.config["ckpt_model_path"]:
                        save_path = os.path.join(os.path.abspath(os.getcwd()),
                                                 self.config["ckpt_model_path"])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        model_save_path = os.path.join(save_path, self.config["model_name"] + ".pkl")
                        torch.save(self.model.state_dict(), model_save_path)
                self.model.train()

    def evaluate(self):
        """验证"""
        self.model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        step = 0
        with torch.no_grad():
            for batch_x, batch_y in self.eval_helper.next_batch(self.eval_inputs, self.eval_labels,
                                                                self.config["batch_size"], self.eval_inputs_len):
                outputs = self.model(batch_x)
                loss = F.cross_entropy(outputs, batch_y)
                loss_total += loss
                true = batch_y.data.cpu().numpy()
                pred = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, true)
                predict_all = np.append(predict_all, pred)
                step += 1

        acc = metrics.accuracy_score(labels_all, predict_all)
        return acc, loss_total / step


if __name__ == '__main__':
    # 读取用户在命令行输入的信息
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config/textcnn.json", help="config path of model")
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
