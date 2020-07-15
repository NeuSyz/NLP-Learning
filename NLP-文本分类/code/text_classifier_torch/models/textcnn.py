import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCnnModel(nn.Module):
    def __init__(self, config, vocab_size=None, word_vectors=None):
        super(TextCnnModel, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors
        self.embedding = nn.Embedding.from_pretrained(self.word_vectors, freeze=False)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.config["num_filters"], (k, self.config["embedding_size"]))
             for k in self.config["filter_sizes"]]
        )
        self.dropout = nn.Dropout(p=self.config["keep_prob"])
        self.fc = nn.Linear(
            self.config["num_filters"] * len(self.config["filter_sizes"]),
            self.config["num_classes"]
        )

    def conv_pooling(self, x, conv):
        """卷积+池化"""
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x, _ = x  # x,seq_len
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_pooling(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
