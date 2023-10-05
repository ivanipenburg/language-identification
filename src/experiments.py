import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from data import get_dataloaders, dict_language_groups
from models import SimpleLSTM, SimpleTransformer
from stream import predict_streaming_batch

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(model, dataloaders, config):
    """
    Function that creates the confusion matrix
    """

    model.to(config['device'])
    model.eval()
    hidden = None

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloaders['test']):
            inputs = batch['input_ids'].to(config['device'])
            labels = batch['label'].to(config['device'])

            logits, hidden = model(inputs.to(config['device']), hidden)
            prediction = torch.argmax(logits, dim=-1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(prediction.cpu().numpy())

    language_groups = dict_language_groups()

    for language_group in language_groups:
        mapped_true_labels = [language_groups[language_group]['mapping_dict'][label] for label in true_labels]
        mapped_predicted_labels = [language_groups[language_group]['mapping_dict'][label] for label in predicted_labels]
        languages = language_groups[language_group]['list']
        labels = np.arange(len(languages))

        # Compute the confusion matrix
        confusion = confusion_matrix(mapped_true_labels, mapped_predicted_labels, labels=labels)

        # Create a ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(confusion, display_labels=languages)

        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Confusion Matrix {language_group}')
        plt.xticks(rotation=90)
        plt.xlabel('Predicted')
        plt.ylabel('True')

        plt.show()

def conf_between_language_families(model, dataloaders, config):

    model.to(config['device'])
    model.eval()
    hidden = None

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloaders['test']):
            inputs = batch['input_ids'].to(config['device'])
            labels = batch['label'].to(config['device'])

            logits, hidden = model(inputs.to(config['device']), hidden)
            prediction = torch.argmax(logits, dim=-1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(prediction.cpu().numpy())

    df = pd.read_csv('data/labels.csv', sep=';')
    language_families = sorted(df['Language family'].unique())
    language_families_mapping = dict()
    for index, row in df.iterrows():
        language_families_mapping[index] = language_families.index(row['Language family'])
    
    mapped_true_labels = [language_families_mapping[label] for label in true_labels]
    mapped_predicted_labels = [language_families_mapping[label] for label in predicted_labels]

    # Compute the confusion matrix
    confusion = confusion_matrix(mapped_true_labels, mapped_predicted_labels, labels=np.arange(len(language_families)))

    # Create a ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion, display_labels=language_families)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix Between Language Families')
    plt.xticks(rotation=90)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()


def most_confused_languages(model, dataloaders, config):
    df = pd.read_csv('data/labels.csv', sep=';')
    dict_language_codes = df.reset_index().set_index('English').to_dict()['index']
    languages = list(dict_language_codes.keys())

    model.to(config['device'])
    model.eval()
    hidden = None

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloaders['test']):
            inputs = batch['input_ids'].to(config['device'])
            labels = batch['label'].to(config['device'])

            logits, hidden = model(inputs.to(config['device']), hidden)
            prediction = torch.argmax(logits, dim=-1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(prediction.cpu().numpy())

    confusion = confusion_matrix(true_labels, predicted_labels, labels=np.arange(len(languages)))

    confusion_counts = {}
    for i in range(len(languages)):
        for j in range(len(languages)):
            if i != j:
                pair = (languages[i], languages[j])
                count = confusion[i][j]
                if pair in confusion_counts:
                    confusion_counts[pair] += count
                else:
                    confusion_counts[pair] = count

    sorted_confusions = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)

    top_30_confused_languages = sorted_confusions[:30]

    for pair, count in top_30_confused_languages:
        language1, language2 = pair
        print(f"{language1} is confused with {language2} - Count: {count}")

    language_pairs, counts = zip(*top_30_confused_languages)
    languages = [f"{language1} with {language2}" for (language1, language2) in language_pairs]

    plt.figure(figsize=(12, 10))
    plt.barh(range(len(languages)), counts, tick_label=languages)
    plt.xlabel('Confusion Count')
    plt.title('Top 30 Most Confused Language Pairs')
    plt.gca().invert_yaxis()  # Invert the y-axis to display the highest confusion at the top
    plt.tight_layout()

    plt.show()


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
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'num_languages': 235,
        'embedding_dim': 128,

        'lstm_hidden_dim': 512,
        'lstm_num_layers': 2,

        'transformer_n_heads': 8,
        'transformer_layers': 2
    }

    if args.dev_mode:
        config['num_languages'] = 4

    dataloaders = get_dataloaders(tokenize_datasets=args.tokenize_datasets,
                                    dev_mode=args.dev_mode, batch_size=256, drop_last=True)

    if config['model'] == 'lstm':
        model = SimpleLSTM(config)
    elif config['model'] == 'transformer':
        model = SimpleTransformer(config)

    model.load_state_dict(torch.load(args.model_checkpoint))

    if args.experiment == 'performance_over_tokens':
        performance_over_tokens(model, dataloaders, config)
    elif args.experiment == 'confusion_matrix':
        plot_confusion_matrix(model, dataloaders, config)
    elif args.experiment == 'most_confused':
        most_confused_languages(model, dataloaders, config)
    elif args.experiment == 'conf_between_groups':
        conf_between_language_families(model, dataloaders, config)
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
        default='src/checkpoints/lstm_19.pt'
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
        action='store_false',
        help='Tokenize dataset (true by default)',
    )

    args = parser.parse_args()
    main(args)