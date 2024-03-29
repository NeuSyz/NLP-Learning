import tensorflow as tf
from .base import BaseModel


class RcnnModel(BaseModel):
    def __init__(self, config, vocab_size, word_vectors):
        super(RcnnModel, self).__init__(config=config, vocab_size=vocab_size, word_vectors=word_vectors)
        self.seq_len = tf.reduce_sum(tf.sign(self.inputs), 1)
        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
        self.init_saver()

    def build_model(self):

        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            init_embeddings = tf.constant_initializer(self.word_vectors)  # 使用预训练词向量初始化
            embedding_w = tf.get_variable("embeddings", shape=[self.vocab_size, self.config["embedding_size"]],
                                          initializer=init_embeddings, trainable=True)
            # if self.word_vectors is not None:
            #     embedding_w = tf.Variable(tf.cast(self.word_vectors, dtype=tf.float32, name="word2vec"),
            #                               name="embedding_w")
            # else:
            #     embedding_w = tf.get_variable("embedding_w", shape=[self.vocab_size, self.config["embedding_size"]],
            #                                   initializer=tf.contrib.layers.xavier_initializer())

            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            embedded_words = tf.nn.embedding_lookup(embedding_w, self.inputs)
            embedded_words_ = embedded_words

        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            for idx, hidden_size in enumerate(self.config["hidden_sizes"]):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=self.keep_prob)
                    # 定义反向LSTM结构
                    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=self.keep_prob)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],
                    # fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs, current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                             lstm_bw_cell,
                                                                             embedded_words,
                                                                             sequence_length=self.seq_len,
                                                                             dtype=tf.float32,
                                                                             scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2]
                    embedded_words = tf.concat(outputs, 2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        fw_output, bw_output = tf.split(embedded_words, 2, -1)

        # 将前向，后向的输出和最早的词向量拼接在一起得到最终的词表征
        with tf.name_scope("word-representation"):
            # [fwOutput, wordEmbedding, bwOutput]
            word_representation = tf.concat([fw_output, embedded_words_, bw_output], axis=2)

        with tf.name_scope("text-representation"):
            output_size = self.config["output_size"]
            # 将拼接后的向量非线性映射到低维得到文本表征
            text_representation = tf.layers.dense(word_representation, output_size, activation=tf.nn.tanh)

        # 每一个位置的值都取所有时序上的最大值，得到最终的特征向量，做max-pool的操作
        output = tf.reduce_max(text_representation, axis=1)

        # 全连接层的输出
        with tf.name_scope("output"):
            output_w = tf.get_variable(
                "outputW",
                shape=[output_size, self.config["num_classes"]],
                initializer=tf.contrib.layers.xavier_initializer())

            output_b = tf.Variable(tf.constant(0.1, shape=[self.config["num_classes"]]), name="output_b")
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            self.logits = tf.nn.xw_plus_b(output, output_w, output_b, name="logits")
            self.predictions = self.get_predictions()

        self.loss = self.cal_loss()
        self.train_op, self.summary_op = self.get_train_op()