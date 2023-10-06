import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm

from data import LANGUAGE_CODES, get_dataloaders
from models import SimpleLSTM, SimpleTransformer
from stream import predict_streaming_batch
from train import evaluate

def plot_confusion_matrix(model, dataloaders, config, languages):
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

            logits, _hidden = model(inputs.to(config['device']))
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

def performance_over_tokens(model, dataloaders, config, groups=None, int_to_label=None):
    """
    Function that plots the performance of the model over the number of tokens
    """
    confidence_per_token = []
    correct_per_token = []
    total_per_token = []

    count = 0

    for batch in tqdm(dataloaders['test']):
        count += 1
        if count > 200:
            break
        predictions, labels, confidences = predict_streaming_batch(batch, model, config)

        for i, prediction in enumerate(predictions):
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


def performance_over_tokens_WORKING(model, dataloaders, config, groups=None, int_to_label=None):
    """
    Function that plots the performance of the model over the number of tokens
    """
    correct_per_token = torch.zeros(size=(200, 128))

    count = 0
    hidden = None

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
                logits, _hidden = model(inpt)
                prediction = torch.argmax(logits, dim=-1)

                if prediction == labels.squeeze():
                    correct_per_token[count-1, i] += 1


    accuracy_per_token = torch.sum(correct_per_token, dim=0) / 200
    plt.plot(range(0, 128), accuracy_per_token)
    plt.xlabel('Number of tokens')
    plt.ylabel('Accuracy')
    plt.show()

# def performance_over_tokens(model, dataloaders, config, groups=None, int_to_label=None):
#     """
#     Function that plots the performance of the model over the number of tokens for different groups.
#     """
#     # Groups is a dict of the form {'Group name': ['lang1', 'lang2', ...], ...}
#     # int_to_label is a dict of the form {0: 'lang1', 1: 'lang2', ...}
#     if groups is None:
#         groups = {'All': LANGUAGE_CODES}
#     if int_to_label is None:
#         int_to_label = {i: code for i, code in enumerate(LANGUAGE_CODES)}

#     count = 0

#     languages = [lang for group in groups for lang in groups[group]]

#     # For each group, have a list of 64 zeroes
#     confidence_per_token = {group: [0 for _ in range(64)] for group in groups}
#     correct_per_token = {group: [0 for _ in range(64)] for group in groups}
#     total_per_token = {group: [0 for _ in range(64)] for group in groups}


#     print(confidence_per_token)

#     for batch in tqdm(dataloaders['test']):
#         # If label is not in the list of any group, skip this batch
#         label = int_to_label[batch['label'].item()]
#         if label not in languages:
#             continue
        

#         count += 1
#         print(count)
#         if count > 500:
#             break
#         predictions, labels, confidences = predict_streaming_batch(batch, model, config)

#         for i, prediction in enumerate(predictions[:64]):
#             for group in groups:
#                 if label in groups[group]:
#                     if prediction == labels:
#                         correct_per_token[group][i] += 1
#                     total_per_token[group][i] += 1
#                     confidence_per_token[group][i] += confidences[i][0][prediction]
#     accuracy_per_token = {group: [correct / total for correct, total in zip(correct_per_token[group], total_per_token[group])] for group in groups}
#     confidence_per_token = {group: [confidence / total for confidence, total in zip(confidence_per_token[group], total_per_token[group])] for group in groups}

#     for group in groups:
#         plt.plot(accuracy_per_token[group], label=group)
#     plt.xlabel('Number of tokens')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.show()

#     for group in groups:
#         plt.plot(confidence_per_token[group], label=group)
#     plt.xlabel('Number of tokens')
#     plt.ylabel('Confidence')
#     plt.legend()
#     plt.show()


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
                                    dev_mode=args.dev_mode, batch_size=1)

    if config['model'] == 'lstm':
        model = SimpleLSTM(config)
    elif config['model'] == 'transformer':
        model = SimpleTransformer(config)

    model.load_state_dict(torch.load(args.model_checkpoint, map_location=config['device']))

    LANGUAGE_GROUPS = {
        'Slavic-Cyrillic': ['rus', 'bul', 'ukr', 'bel', 'mkd'],
        'Old Norse': ['swe', 'dan', 'nor', 'isl', 'fao'],
        'West Germanic': ['eng', 'deu', 'nld', 'afr', 'nds'],
        'Sino-Tibetan': ['cmn', 'yue', 'wuu', 'bod', 'hak'],
    }

    TEMP_GROUP = {
        'Dutch': ['nld'],
        'English': ['eng'],
        'German': ['deu'],
        'French': ['fra'],
    }

    SLAVIC_CYRILLIC_GROUPS = {
        'Russian': ['rus'],
        'Bulgarian': ['bul'],
        'Ukrainian': ['ukr'],
        'Belarusian': ['bel'],
        'Macedonian': ['mkd'],
    }

    OLD_NORSE_GROUPS = {
        'Swedish': ['swe'],
        'Danish': ['dan'],
        'Norwegian': ['nor'],
        'Icelandic': ['isl'],
        'Faroese': ['fao'],
    }

    WEST_GERMANIC_GROUPS = {
        'English': ['eng'],
        'German': ['deu'],
        'Dutch': ['nld'],
        'Afrikaans': ['afr'],
        'Low German': ['nds'],
    }

    SINO_TIBETAN_GROUPS = {
        'Mandarin': ['cmn'],
        'Cantonese': ['yue'],
        'Wu': ['wuu'],
        'Tibetan': ['bod'],
        'Hakka': ['hak'],
    }

    if args.experiment == 'performance_over_tokens':
        performance_over_tokens(model, dataloaders, config, TEMP_GROUP, int_to_label)
    elif args.experiment == 'confusion_matrix':
        assert args.tokenize_datasets, 'Need a tokenized dataset, use flag --tokenize_datasets'
        plot_confusion_matrix(model, dataloaders, config, languages)
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

    args = parser.parse_args()
    main(args)