import torch
import torch.nn as nn
import torch.nn.functional as F


class RCnnModel(nn.Module):
    def __init__(self, config, vocab_size=None, word_vectors=None):
        super(RCnnModel, self).__init__()
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
        self.max_pooling = nn.MaxPool1d(self.config["sequence_length"])
        self.fc = nn.Linear(
            self.config["hidden_size"] * 2 + self.config["embedding_size"],
            self.config["num_classes"]
        )

    def forward(self, x):
        x, seq_len = x
        embed = self.embedding(x)  # [batch_size, seq_len, embedding]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)  # [batch_size, seq_len, hidden_size*2+embedding]
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.max_pooling(out).squeeze()
        out = self.fc(out)
        return out
