import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from data import get_dataloaders
from models import SimpleLSTM, SimpleTransformer, TextCNN

import argparse


CHECKPOINT_PATH = "checkpoints/"

def evaluate(dataloaders, model, config):
    with torch.no_grad():
        correct = 0
        total = 0

        for batch in tqdm(dataloaders['test']):
            labels = batch['label'].to(config['device'])
            logits, _ = model(batch['input_ids'].to(config['device']))
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
            logits, _ = model(batch['input_ids'].to(config['device']))
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch} loss: {loss.item()}')

        evaluate(dataloaders, model, config)

        torch.save(model.state_dict(), f'{CHECKPOINT_PATH}model_{epoch}.pt')

def main(args):
    config = {
        'model': args.model,
        'lr': args.lr,
        'epochs': args.epochs,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'num_languages': 235,
        'hidden_dim': 128,
        'embedding_dim': 100,
    }

    dataloaders = get_dataloaders(tokenize_datasets=args.tokenize_datasets,
                                  dev_mode=args.dev_mode)

    if args.dev_mode:
        config['num_languages'] = 4

    if config['model'] == 'lstm':
        model = SimpleLSTM(config)
    elif config['model'] == 'transformer':
        model = SimpleTransformer(config)
    else:
        raise NotImplementedError('Model is not implemented!')

    print(f"Selected {config['model']} model...")
    model.to(config['device'])

    train(dataloaders, model, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language Identification',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--model",
        type=str,
        help="Model to use",
        choices=['lstm', 'transformer'],
        default='lstm'
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
        default=1e-3
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs",
        default=5
    )

    parser.add_argument(
        "--tokenize_datasets",
        action="store",
        help="Tokenize dataset (true by default)",
        default=True
    )

    parser.add_argument(
        "--train_bpe",
        action="store",
        help="Train BPE tokenizer (true by default)",
        default=True
    )

    parser.add_argument(
        "--dev_mode",
        action="store_true",
        help="Dev mode (off by default) reduces amount of languages to 4"
    )

    args = parser.parse_args()
    main(args)