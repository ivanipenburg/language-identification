import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm

from data import LANGUAGE_CODES, get_dataloaders
from models import SimpleLSTM, SimpleTransformer

LANGUAGE_GROUPS = {
    'Slavic-Cyrillic': ['rus', 'bul', 'ukr', 'bel', 'mkd'],
    'Old Norse': ['swe', 'dan', 'nor', 'isl', 'fao'],
    'West Germanic': ['eng', 'deu', 'nld', 'afr', 'nds'],
    'Sino-Tibetan': ['cmn', 'yue', 'wuu', 'bod', 'hak'],
}

def plot_confusion_matrix(model, dataloaders, config, languages):
    """
    Function that creates the confusion matrix
    """

    model.to(config['device'])
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloaders['test']):
            inputs = batch['input_ids'].to(config['device'])
            labels = batch['label'].to(config['device'])

            logits, _ = model(inputs.to(config['device']))
            prediction = torch.argmax(logits, dim=-1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(prediction.cpu().numpy())

    # Compute the confusion matrix
    confusion = confusion_matrix(true_labels, predicted_labels, labels=np.arange(len(languages)))

    # Create a ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion, display_labels=languages)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def performance_over_thresholds(model, dataloaders, config):
    '''
    Plot the performance over different thresholds
    '''
    thresholds = np.arange(0.1, 1, 0.1)

    model = model.to(config['device'])
    model.eval()

    accuracies = []
    number_of_tokens_for_prediction = []

    with torch.no_grad():
        for threshold in thresholds:
            print(f'Threshold: {threshold}')
            correct = 0
            count = 0
            num_tokens = []

            for batch in tqdm(dataloaders['test']):
                if count >= 1000:
                    break
                count += 1
                
                labels = batch['label'].to(config['device'])
                inputs = batch['input_ids'].to(config['device'])

                for i in range(0, 128):
                    inpt = inputs[:,:i+1]
                    logits, _ = model(inpt)
                    prediction = torch.argmax(logits, dim=-1)
                    confidence = torch.softmax(logits, dim=-1)

                    if confidence[0][prediction][0] > threshold:
                        if prediction == labels.squeeze():
                            correct += 1
                        break
                    num_tokens.append(i)

            accuracies.append(correct / count)
            number_of_tokens_for_prediction.append(np.mean(num_tokens))

    plt.plot(thresholds, accuracies)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(thresholds, number_of_tokens_for_prediction)
    plt.xlabel('Threshold')
    plt.ylabel('Number of tokens for prediction')
    plt.show()


def performance_over_tokens(model, dataloaders, config, groups=None, int_to_label=None):
    """
    Function that plots the performance of the model over the number of tokens
    """
    correct_per_token = torch.zeros(size=(200, 128))

    count = 0

    model = model.to(config['device'])
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloaders['test']):
            if count >= 200:
                break

            count += 1

            labels = batch['label'].to(config['device'])
            inputs = batch['input_ids'].to(config['device'])

            for i in range(0, 128):
                inpt = inputs[:,:i+1]
                logits, _ = model(inpt)
                prediction = torch.argmax(logits, dim=-1)

                if prediction == labels.squeeze():
                    correct_per_token[count-1, i] += 1


    accuracy_per_token = torch.sum(correct_per_token, dim=0) / 200
    plt.plot(range(0, 128), accuracy_per_token)
    plt.xlabel('Number of tokens')
    plt.ylabel('Accuracy')
    plt.show()

def performance_over_tokens_per_language_group(model, dataloaders, config, groups=None, int_to_label=None):
    num_examples = 200
    num_groups = len(groups)
    correct_per_token = torch.zeros(size=(num_groups, num_examples, 128))
    confidence_per_token = torch.zeros(size=(num_groups, num_examples, 128))

    group_count = {group: 0 for group in groups}

    print(group_count)

    model = model.to(config['device'])
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloaders['test']):
            label = int_to_label[batch['label'].item()]
            if not any(label in group for group in groups.values()):
                continue

            # Get current group
            group = [group for group in groups if label in groups[group]][0]
            group_num = list(groups.keys()).index(group)

            # Skip if we have already seen num_examples samples for this group
            label = int_to_label[batch['label'].item()]
            if group_count[group] >= num_examples:
                continue

            # break if we have seen num_examples samples for each group
            if all(group_count[group] >= num_examples for group in groups):
                break

            group_count[group] += 1
            print(group_count)

            labels = batch['label'].to(config['device'])
            inputs = batch['input_ids'].to(config['device'])

            for i in range(0, 128):
                inpt = inputs[:,:i+1]
                logits, _ = model(inpt)
                prediction = torch.argmax(logits, dim=-1) 
                confidence = torch.softmax(logits, dim=-1)

                if prediction == labels.squeeze():
                    correct_per_token[group_num, group_count[group]-1, i] += 1
                confidence_per_token[group_num, group_count[group]-1, i] += confidence[0][prediction][0]

    accuracy_per_token = torch.sum(correct_per_token, dim=1) / num_examples
    confidence_per_token = torch.sum(confidence_per_token, dim=1) / num_examples

    for i, group in enumerate(groups):
        plt.plot(range(1, 128), accuracy_per_token[i][1:], label=group)

    plt.xlabel('Number of tokens')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    for i, group in enumerate(groups):
        plt.plot(range(1, 128), confidence_per_token[i][1:], label=group)

    plt.xlabel('Number of tokens')
    plt.ylabel('Confidence')
    plt.legend()
    plt.show()


def main(args):

    config = {
        'model': args.model,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'num_languages': 235,
        'embedding_dim': 128,

        'lstm_hidden_dim': 512,
        'lstm_num_layers': 2,

        'transformer_n_heads': 8,
        'transformer_layers': 2
    }

    print(vars(args))

    if args.dev_mode:
        config['num_languages'] = 4
        languages = ['eng', 'deu', 'fra', 'nld']
    else:
        languages = LANGUAGE_CODES

    dataloaders, int_to_label = get_dataloaders(tokenize_datasets=args.tokenize_datasets,
                                    dev_mode=args.dev_mode, batch_size=args.batch_size)

    if config['model'] == 'lstm':
        model = SimpleLSTM(config)
    elif config['model'] == 'transformer':
        model = SimpleTransformer(config)

    model.load_state_dict(torch.load(args.model_checkpoint, map_location=config['device']))



    if args.experiment == 'performance_over_tokens':
        if args.batch_size != 1:
            print('Batch size must be 1 for this experiment!')
            return
        performance_over_tokens(model, dataloaders, config)
    elif args.experiment == 'performance_over_tokens_per_language_group':
        if args.batch_size != 1:
            print('Batch size must be 1 for this experiment!')
            return
        performance_over_tokens_per_language_group(model, dataloaders, config, LANGUAGE_GROUPS, int_to_label)
    elif args.experiment == 'confusion_matrix':
        assert args.tokenize_datasets, 'Need a tokenized dataset, use flag --tokenize_datasets'
        plot_confusion_matrix(model, dataloaders, config, languages)
    elif args.experiment == 'thresholds':
        performance_over_thresholds(model, dataloaders, config)
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
        default='src/checkpoints/model_4.pt'
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

    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size',
        default=1
    )

    args = parser.parse_args()
    main(args)