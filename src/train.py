import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from data import get_dataloaders
from model import SimpleBiLSTM


def evaluate(dataloaders, model, config):
    with torch.no_grad():
        correct = 0
        total = 0

        for batch in tqdm(dataloaders['test']):
            labels = batch['label'].to(config['device'])
            logits = model(batch['input_ids'])
            predictions = torch.argmax(logits, dim=-1)
            correct += torch.sum(predictions == labels)
            total += predictions.shape[0]

        print(f'Test accuracy: {correct / total}')


def train(dataloaders, model, config):
    print('Starting training...')
    optimizer = Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(config['epochs'])):
        for batch in tqdm(dataloaders['train']):
            optimizer.zero_grad()
            labels = batch['label'].to(config['device'])
            logits = model(batch['input_ids'])
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch} loss: {loss.item()}')

        evaluate(dataloaders, model, config)


if __name__ == '__main__':
    config = {
        'lr': 1e-3,
        'epochs': 10,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

    model = SimpleBiLSTM()
    model.to(config['device'])

    dataloaders = get_dataloaders(tokenize_datasets=True, dev_mode=True)
    train(dataloaders, model, config)