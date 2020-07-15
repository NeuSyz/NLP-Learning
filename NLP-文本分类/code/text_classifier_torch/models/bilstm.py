import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLstmModel(nn.Module):
    def __init__(self, config, vocab_size=None, word_vectors=None):
        super(BiLstmModel, self).__init__()
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
        self.fc = nn.Linear(self.config["hidden_size"] * 2, self.config["num_classes"])

    def forward(self, x):
        x, seq_len = x
        out = self.embedding(x)
        # 采用变长RNN
        _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从大到小排序后index
        _, idx_unsort = torch.sort(idx_sort)  # 原序列index
        out = torch.index_select(out, 0, idx_sort)
        seq_len = list(seq_len[idx_sort])
        out = pack_padded_sequence(out, seq_len, batch_first=True)
        # out:[batch_size, seq_len, 2 * hidden_size] hn:[num_layer*2,batch,hidden_size]
        out, (hn, cn) = self.lstm(out)
        # 前向后向拼接
        out = torch.cat((hn[0], hn[1]), -1)
        # out, seq_len = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = out.index_select(0, idx_unsort)
        out = self.fc(out)
        return out

    # def forward(self, x):
    #     x, _ = x
    #     out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
    #     out, _ = self.lstm(out)
    #     out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
    #     return out
