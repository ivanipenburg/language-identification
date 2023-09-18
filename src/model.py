import torch
import torch.nn as nn
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
num_tokens = tokenizer.vocab_size

NUMBER_OF_CLASSES = 4
EMBEDDING_DIM = 300
HIDDEN_DIM = 512


class SimpleBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, EMBEDDING_DIM)
        self.lstm = nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_DIM, num_layers=1, bidirectional=True)
        self.classifier = nn.Linear(HIDDEN_DIM * 2, NUMBER_OF_CLASSES)
        nn.init.xavier_normal_(self.classifier.weight)
        
    def forward(self, input_ids):
        input_ids = self.embedding(input_ids)
        outputs, _ = self.lstm(input_ids)
        outputs = torch.cat((outputs[:, -1, :HIDDEN_DIM], outputs[:, 0, HIDDEN_DIM:]), dim=-1)
        logits = self.classifier(outputs)
        return logits
    
    