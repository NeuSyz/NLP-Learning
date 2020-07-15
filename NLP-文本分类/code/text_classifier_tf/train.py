import json
import os
import argparse
import tensorflow as tf
from models import TextCnnModel, BiLstmModel, BiLstmAttenModel, RcnnModel, TransformerModel, HANModel
from utils.metrics import get_binary_metrics, get_multi_metrics, mean
from utils.data_helper import DataHelper


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        with open(os.path.join(os.getcwd(), args.config_path), "r") as fr:
            self.config = json.load(fr)

        self.train_helper = None
        self.eval_helper = None
        self.model = None
        self.current_step = 0

        # 加载数据集
        self.load_data()
        self.train_inputs, self.train_labels = self.train_helper.gen_data()
        print("train data size: {}".format(len(self.train_labels)))
        self.vocab_size = len(self.train_helper.word2id)
        print("vocab size: {}".format(self.vocab_size))
        self.word_vectors = self.train_helper.word2vec
        self.label_list = [value for key, value in self.train_helper.label2id.items()]

        self.eval_inputs, self.eval_labels = self.eval_helper.gen_data()
        print("eval data size: {}".format(len(self.eval_labels)))
        print("label numbers: ", len(self.label_list))
        # 初始化模型对象
        self.create_model()

    def load_data(self):
        """
        创建数据对象
        :return:
        """
        # 生成训练集对象
        self.train_helper = DataHelper(self.config, train=True)

        # 生成验证集对象
        self.eval_helper = DataHelper(self.config, train=False)

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        if self.config["model_name"] == "textcnn":
            self.model = TextCnnModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        elif self.config["model_name"] == "bilstm":
            self.model = BiLstmModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        elif self.config["model_name"] == "bilstm_atten":
            self.model = BiLstmAttenModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        elif self.config["model_name"] == "rcnn":
            self.model = RcnnModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        elif self.config["model_name"] == "transformer":
            self.model = TransformerModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        elif self.config["model_name"] == "han":
            self.model = HANModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)

    def train(self):
        """
        训练模型
        :return:
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            # 初始化变量值
            sess.run(tf.global_variables_initializer())
            current_step = 0

            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))

                for batch in self.train_helper.next_batch(self.train_inputs, self.train_labels,
                                                            self.config["batch_size"]):
                    summary, loss, predictions = self.model.train(sess, batch, self.config["keep_prob"])

                    if current_step % 5 == 0:
                        print("train: step: {}, loss: {}".format(current_step, loss))

                    current_step += 1
                    if self.eval_helper and current_step % self.config["eval_every"] == 0:

                        eval_losses = []
                        eval_accs = []
                        eval_aucs = []
                        eval_recalls = []
                        eval_precs = []
                        eval_f_betas = []
                        for eval_batch in self.eval_helper.next_batch(self.eval_inputs, self.eval_labels,
                                                                        self.config["batch_size"]):
                            eval_summary, eval_loss, eval_predictions = self.model.eval(sess, eval_batch)

                            eval_losses.append(eval_loss)
                            if self.config["num_classes"] == 1:
                                acc, auc, recall, prec, f_beta = get_binary_metrics(pred_y=eval_predictions,
                                                                                    true_y=eval_batch["y"])
                                eval_accs.append(acc)
                                eval_aucs.append(auc)
                                eval_recalls.append(recall)
                                eval_precs.append(prec)
                                eval_f_betas.append(f_beta)
                            elif self.config["num_classes"] > 1:
                                acc, recall, prec, f_beta = get_multi_metrics(pred_y=eval_predictions,
                                                                              true_y=eval_batch["y"],
                                                                              labels=self.label_list)
                                eval_accs.append(acc)
                                eval_recalls.append(recall)
                                eval_precs.append(prec)
                                eval_f_betas.append(f_beta)
                        print("\n")
                        print("eval:  loss: {}, acc: {}, auc: {}, recall: {}, precision: {}, f_beta: {}".format(
                            mean(eval_losses), mean(eval_accs), mean(eval_aucs), mean(eval_recalls),
                            mean(eval_precs), mean(eval_f_betas)))
                        print("\n")

                        if self.config["ckpt_model_path"]:
                            save_path = os.path.join(os.getcwd(), self.config["ckpt_model_path"])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            self.model.saver.save(sess, model_save_path, global_step=current_step)


if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config/bilstm.json", help="config path of model")
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
