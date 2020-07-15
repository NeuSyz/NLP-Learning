import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLstmAttenModel(nn.Module):
    def __init__(self, config, vocab_size=None, word_vectors=None):
        super(BiLstmAttenModel, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors
        self.embedding = nn.Embedding.from_pretrained(self.word_vectors, freeze=False)
        self.lstm = nn.LSTM(
            self.config["embedding_size"],
            self.config["hidden_size"],
            self.config["num_layers"],
            bidirectional=True,  # 双向默认已是2层，final_num_layer=2*num_layers
            batch_first=True,
            dropout=self.config["keep_prob"]  # 加在堆叠RNN的层与层之间，最后一层没有
        )
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.Tensor(self.config["hidden_size"] * 2))
        self.fc = nn.Linear(self.config["hidden_size"] * 2, self.config["num_classes"])

    def forward(self, x):
        x, seq_len = x
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 100, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[64, 100, 512]

        # 计算attention scores
        M = self.tanh1(H)  # [64, 100 512]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [64, 100, 1]
        out = H * alpha  # [64, 100, 512]
        out = torch.sum(out, 1)  # [64, 512]

        out = self.fc(out)
        return out
