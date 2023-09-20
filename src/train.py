import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from data import get_dataloaders
from models import SimpleLSTM, TextCNN, SimpleTransformer

CHECKPOINT_PATH = "/media/jessebelleman/DATA2/checkpoints/"

def evaluate(dataloaders, model, config):
    with torch.no_grad():
        correct = 0
        total = 0

        for batch in tqdm(dataloaders['test']):
            labels = batch['label'].to(config['device'])
            logits = model(batch['input_ids'].to(config['device']))
            predictions = torch.argmax(logits, dim=-1)
            correct += torch.sum(predictions == labels)
            total += predictions.shape[0]

        print(f'Test accuracy: {correct / total}')


def train(dataloaders, model, config):
    print('Starting training...')
    optimizer = Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(config['epochs']), desc='Epochs'):
        for batch in tqdm(dataloaders['train'], desc='Iterations'):
            optimizer.zero_grad()
            labels = batch['label'].to(config['device'])
            logits = model(batch['input_ids'].to(config['device']))
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch} loss: {loss.item()}')

        evaluate(dataloaders, model, config)

        torch.save(model.state_dict(), f'{CHECKPOINT_PATH}model_{epoch}.pt')


if __name__ == '__main__':
    config = {
        'model': 'transformer',
        'lr': 1e-3,
        'epochs': 20,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'num_languages': 235,
        'hidden_dim': 64,
        'embedding_dim': 100,
    }

    dev_mode = True
    dataloaders = get_dataloaders(tokenize_datasets=True, dev_mode=dev_mode)

    if dev_mode:
        config['num_languages'] = 4

    if config['model'] == 'lstm':
        model = SimpleLSTM(config)
    elif config['model'] == 'transformer':
        model = SimpleTransformer(config)
    else:
        raise NotImplementedError("Model is not implemented, check config!")

    print(f"Selected {config['model']} model...")
    model.to(config['device'])

    train(dataloaders, model, config)