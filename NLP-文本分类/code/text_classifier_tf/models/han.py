import tensorflow as tf
from .base import BaseModel


class HANModel(BaseModel):
    """Hierarchical Attention Network 分级注意力网络 适合长文本分类"""
    def __init__(self, config, vocab_size, word_vectors):
        super(HANModel, self).__init__(config=config, vocab_size=vocab_size, word_vectors=word_vectors)
        self.seq_len = tf.reduce_sum(tf.sign(self.inputs), 1)
        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
        self.init_saver()

    def build_model(self):

        # 词嵌入层
        with tf.name_scope("embedding"):
            # 模型按照句子长度将文本分句，实际可以采用标点符号进行分句，理论效果可能好
            input_x = tf.split(self.inputs, self.config["num_split_sentence"], axis=1)
            # [batch_size, num_sent, sent_len] sent_len=seq_len/num_sent
            input_x = tf.stack(input_x, axis=1)

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

            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch,num_sent,sent_len,embed_size]
            embedded_words = tf.nn.embedding_lookup(embedding_w, input_x)
            sent_len = int(self.config["sequence_length"] / self.config["num_split_sentence"])
            # [batch*num_sent,sent_len,embed_size]
            embedded_words = tf.reshape(embedded_words, shape=[-1, sent_len, self.config["embedding_size"]])

        # 计算得到词级别隐向量
        with tf.name_scope("word-vec"):
            output_fw, output_bw = self._Bidirectional_Encoder(embedded_words, "word_vec")
            # [batch_size*num_sentences,sentence_length,hidden_size * 2]
            word_hidden_state = tf.concat((output_fw, output_bw), 2)
        # 给词级别隐向量加入attention得到句子表征
        with tf.name_scope("word_attention"):
            # [batch*num_sent,hidden_size * 2]
            sentence_vec = self._attention(word_hidden_state, "word_attention")
        # 计算得到句子级别隐向量
        with tf.name_scope("sentence_vec"):
            # [batch_size,num_sentences,hidden_size*2]
            sentence_vec = tf.reshape(sentence_vec, shape=[-1, self.config["num_split_sentence"],
                                                           self.config["context_dim"] * 2])
            output_fw, output_bw = self._Bidirectional_Encoder(sentence_vec, "sentence_vec")
            # [batch_size*num_sentences,sentence_length,hidden_size * 2]
            sentence_hidden_state = tf.concat((output_fw, output_bw), 2)
        # 给句子级别隐向量加入attention得到整个文档向量
        with tf.name_scope("sentence_attention"):
            # [batch_size, hidden_size * 2]
            doc_vec = self._attention(sentence_hidden_state, "sentence_attention")

        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(doc_vec, self.keep_prob)
            output_size = h_drop.get_shape()[-1].value

        # 全连接层的输出
        with tf.name_scope("output"):
            output_w = tf.get_variable(
                "outputW",
                shape=[output_size, self.config["num_classes"]],
                initializer=tf.contrib.layers.xavier_initializer())

            output_b = tf.Variable(tf.constant(0.1, shape=[self.config["num_classes"]]), name="output_b")
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            self.logits = tf.nn.xw_plus_b(h_drop, output_w, output_b, name="logits")
            self.predictions = self.get_predictions()

        self.loss = self.cal_loss()
        self.train_op, self.summary_op = self.get_train_op()

    def _get_cell(self):
        """rnn中神经单元的选择"""
        if self.config["rnn_type"] == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(num_units=self.config["context_dim"])
        elif self.config["rnn_type"] == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config["context_dim"])
        else:
            return tf.nn.rnn_cell.GRUCell(num_units=self.config["context_dim"])

    def _Bidirectional_Encoder(self, inputs, name):
        """单层双向RNN"""
        with tf.variable_scope(name):
            # 定义前向rnn结构
            fw_cell = self._get_cell()
            rnn_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                fw_cell,
                output_keep_prob=self.keep_prob)
            # 定义反向rnn结构
            bw_cell = self._get_cell()
            rnn_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                bw_cell,
                output_keep_prob=self.keep_prob)

            # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
            # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],
            # fw和bw的hidden_size一样
            # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
            outputs, current_state = tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell,
                                                                     rnn_bw_cell,
                                                                     inputs,
                                                                     # sequence_length=self.seq_len,
                                                                     dtype=tf.float32
                                                                     )

            # [batch_size, time_step, hidden_size]
            output_fw, output_bw = outputs
            return output_fw, output_bw

    def _attention(self, inputs, name):
        with tf.variable_scope(name):
            # 使用一个全连接层编码 GRU 的输出，相当于一个隐藏层
            # [batch_size,sentence_length,hidden_size * 2]
            hidden_vec = tf.layers.dense(inputs, self.config["hidden_size"] * 2,
                                         activation=tf.nn.tanh, name='w_hidden')

            # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
            # [hidden_size * 2]
            u_context = tf.Variable(tf.truncated_normal([self.config["hidden_size"] * 2]), name='u_context')
            # [batch_size,sequence_length]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(hidden_vec, u_context),
                                                axis=2, keep_dims=True), dim=1)
            # before reduce_sum [batch_size, sequence_length, hidden_szie*2]，
            # after reduce_sum [batch_size, hidden_size*2]
            attention_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)

        return attention_output
