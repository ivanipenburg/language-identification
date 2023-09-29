import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from data import get_dataloaders
from models import SimpleLSTM, SimpleTransformer
from stream import predict_streaming_batch


def performance_over_tokens(model, dataloaders, config):
    """
    Function that plots the performance of the model over the number of tokens
    """
    confidence_per_token = []
    correct_per_token = []
    total_per_token = []

    for batch in tqdm(dataloaders['test']):
        predictions, labels, confidences = predict_streaming_batch(batch, model, config)

        for i, prediction in enumerate(predictions[:64]):
            # Make correct_per_token 2d with first dimension being the number of tokens
            # and the second dimension being the number of correct predictions
            if i >= len(correct_per_token):
                correct_per_token.append([])
                total_per_token.append([])
                confidence_per_token.append([])
            if prediction == labels:
                correct_per_token[i].append(1)
            else:
                correct_per_token[i].append(0)

            total_per_token[i].append(1)
            confidence_per_token[i].append(confidences[i][0][prediction])

    accuracy_per_token = [np.sum(correct) / np.sum(total) for correct, total in zip(correct_per_token, total_per_token)]
    confidence_per_token = [np.mean(confidence) for confidence in confidence_per_token]

    plt.plot(accuracy_per_token)
    plt.xlabel('Number of tokens')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(confidence_per_token)
    plt.xlabel('Number of tokens')
    plt.ylabel('Confidence')
    plt.show()


def main(args):
    config = {
        'model': args.model,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_languages': 235,
        'hidden_dim': 128,
        'embedding_dim': 100,
    }

    if args.dev_mode:
        config['num_languages'] = 4

    dataloaders = get_dataloaders(tokenize_datasets=args.tokenize_datasets,
                                    dev_mode=args.dev_mode, batch_size=1)

    if config['model'] == 'lstm':
        model = SimpleLSTM(config)
    elif config['model'] == 'transformer':
        model = SimpleTransformer(config)

    model.load_state_dict(torch.load(args.model_checkpoint))

    if args.experiment == 'performance_over_tokens':
        performance_over_tokens(model, dataloaders, config)
    elif args.experiment == 'test':
        print('Running testing experiment...')
    elif args.experiment == 'train_test':
        print('Running train-test experiment...')
    else:
        raise NotImplementedError('Experiment is not implemented!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language Identification',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--model',
        type=str,
        help='Model to use',
        choices=['lstm', 'transformer'],
        default='lstm'
    )

    parser.add_argument(
        '--model_checkpoint',
        type=str,
        help='Model checkpoint to use',
        default='checkpoints/model_4.pt'
    )

    parser.add_argument(
        '--experiment',
        type=str,
        help="Experiment to run",
    )

    parser.add_argument(
        '--dev_mode',
        action='store_true',
        help='Run in development mode',
    )

    parser.add_argument(
        '--tokenize_datasets',
        action='store_true',
        help='Tokenize dataset (true by default)',
    )

    args = parser.parse_args()
    main(args)