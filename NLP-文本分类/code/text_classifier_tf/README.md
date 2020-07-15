### 文本分类
基于tf 1.x 的多种baseline模型的进行单标签文本分类任务(支持二分类和多分类)，模型包括：**textCNN、Bi-LSTM、Bi-LSTM+Attention、HAN(多层注意力网络)、RCNN、Transformer**。

#### 环境要求
python = 3.6
tensorflow = 1.12
scikit-learn

#### 数据集
采用竞赛——O2O商铺食品安全相关评论发现数据集

#### 文件结构介绍
* config文件：配置各种模型的配置参数
* data：存放训练集和测试集及预处理后文件等
* ckpt_model：存放checkpoint模型文件
* utils：提供数据处理的方法
* models：存放模型代码
* train.py：模型训练
* predict.py：模型预测
* test.py：模型测试

#### 数据预处理
要求训练集和测试集分开存储，对于中文的数据必须先分词，对分词后的词用空格符分开，并且将标签连接到每条数据的尾部，标签和句子用分隔符\<SEP>分开。具体的如下：
* 今天 的 天气 真好\<SEP>积极

具体做法：
在utils目录下，data_processor.py设计数据预处理类。
注意：utils.py中设置自定义数据路径与参数；wv目录下的预训练词向量文件自行准备。

#### 训练模型
例如
`python train.py --config_path="config/textcnn.json"`

#### 预测模型
预测代码都在predict.py中，初始化Predictor对象，调用predict方法即可。

#### 模型的配置参数详述
主要介绍config目录下配置文件基本参数：

* model_name：模型名称
* epochs：全样本迭代次数
* checkpoint_every：迭代多少步保存一次模型文件
* eval_every：迭代多少步验证一次模型
* learning_rate：学习速率
* optimization：优化算法
* embedding_size：embedding层大小
* batch_size：批样本大小
* sequence_length：序列长度
* num_classes：样本的类别数，二分类时置为1，多分类时置为实际类别数
* keep_prob：保留神经元的比例
* l2_reg_lambda：L2正则化的系数，主要对全连接层的参数正则化
* max_grad_norm：梯度阶段临界值
* data_path：预处理数据保存路径
* output_path：输出路径，用来存储测试数据结果
* ckpt_model_path：checkpoint 模型的存储路径

