import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
num_tokens = tokenizer.vocab_size

def get_parameter_count(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([p.numel() for p in model_parameters])

class SimpleLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, config['embedding_dim'])
        self.lstm = nn.LSTM(input_size=config['embedding_dim'],
                            hidden_size=config['lstm_hidden_dim'],
                            num_layers=config['lstm_num_layers'], batch_first=True)
        self.classifier = nn.Linear(config['lstm_hidden_dim'], config['num_languages'])

    def forward(self, input_ids, hidden=None):
        embeddings = self.embedding(input_ids)
        lstm_out, hidden = self.lstm(embeddings, hidden)
        lstm_out = torch.max(lstm_out, dim=1)[0]
        output = self.classifier(lstm_out)
        return output, hidden


class SimpleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, config['embedding_dim'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=config['embedding_dim'],
                                                   nhead=config['transformer_n_heads'],
                                                   dim_feedforward=4096,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=config['transformer_layers'])
        self.classifier = nn.Linear(config['embedding_dim'], config['num_languages'])

    def forward(self, input_ids, hidden=None):
        input_ids = self.embedding(input_ids)
        # input_ids = input_ids.permute(1, 0, 2)

        if hidden is not None:
            outputs = self.encoder(input_ids, src_key_padding_mask=hidden)
        else:
            outputs = self.encoder(input_ids)
    
        # outputs = outputs.permute(1, 0, 2)
        outputs = torch.max(outputs, dim=1)[0]
        logits = self.classifier(outputs)
        return logits, hidden


class TextCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, config['embedding_dim'])
        self.conv1 = nn.Conv1d(config['embedding_dim'], 128, kernel_size=3)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3)
        self.classifier = nn.Linear(128, config['num_languages'])

    def forward(self, input_ids):
        input_ids = self.embedding(input_ids)
        input_ids = input_ids.permute(0, 2, 1)
        outputs = F.relu(self.conv1(input_ids))
        outputs = F.relu(self.conv2(outputs))
        outputs = F.relu(self.conv3(outputs))
        outputs = torch.max(outputs, dim=-1)[0]
        logits = self.classifier(outputs)
        return logits
