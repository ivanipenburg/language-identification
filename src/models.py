import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
num_tokens = tokenizer.vocab_size

class SimpleLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, config['embedding_dim'])
        self.lstm = nn.LSTM(input_size=config['embedding_dim'], hidden_size=config['hidden_dim'], num_layers=1, batch_first=True)
        self.classifier = nn.Linear(config['hidden_dim'], config['num_languages'])

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = torch.max(lstm_out, dim=1)[0]
        output = self.classifier(lstm_out)
        return output


class SimpleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, config['embedding_dim'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=config['embedding_dim'], nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        nn.Transformer(d_model=config['embedding_dim'], nhead=4, num_encoder_layers=2, num_decoder_layers=2, batch_first=True)
        self.classifier = nn.Linear(config['embedding_dim'], config['num_languages'])

    def forward(self, input_ids):
        input_ids = self.embedding(input_ids)
        input_ids = input_ids.permute(1, 0, 2)
        outputs = self.encoder(input_ids)
        outputs = outputs.permute(1, 0, 2)
        outputs = torch.max(outputs, dim=1)[0]
        logits = self.classifier(outputs)
        return logits


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
