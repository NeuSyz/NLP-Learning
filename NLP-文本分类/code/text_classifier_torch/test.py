import json
import pandas as pd
from predict import Predictor
import argparse
import os


class TestProcessor(object):
    def __init__(self, args):
        self.args = args
        with open(os.path.join(os.path.abspath(os.getcwd()), args.config_path), "r") as fr:
            self.config = json.load(fr)
        self.predictor = Predictor(self.config)
        self._test_data_path = os.path.join(os.getcwd(), '{}/test.txt'.format(self.config["data_path"]))
        self._output_path = os.path.join(os.getcwd(), self.config["output_path"])
        self.inputs, self.data_ids = self.read_data()

    def read_data(self):
        """
        读取数据
        :return: 返回分词后的文本内容和标签，inputs = [[]], labels = []
        """
        inputs = []
        data_ids = []
        with open(self._test_data_path, "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                try:
                    text, data_id = line.strip().split("<SEP>")
                    inputs.append(text.strip().split(" "))
                    data_ids.append(data_id.strip())
                except:
                    print("error")
        return inputs, data_ids

    @ staticmethod
    def next_batch(x, batch_size=128):
        """
        分批测试数据
        :param x:
        :param batch_size:
        :return:
        """
        num_batches = (len(x) - 1) // batch_size + 1

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(x))
            batch_x = x[start: end]
            yield batch_x

    def gen_csv(self):
        """生成预测结果csv文件"""
        predictions = []
        step = 0
        for batch_x in self.next_batch(self.inputs):
            step += 1
            print("test: step {}".format(step))
            batch_predict = self.predictor.predict(batch_x)
            predictions.extend(batch_predict)

        df = pd.DataFrame([self.data_ids, predictions])
        df = df.T.rename(columns={0: 'id', 1: 'label'})
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)
        df.to_csv(self._output_path + '/{}.csv'.format(self.config["model_name"]),
                  sep=',', encoding='utf-8', index=False)
        print("create .csv finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config/textcnn.json", help="config path of model")
    args = parser.parse_args()
    test_processor = TestProcessor(args)
    test_processor.gen_csv()
