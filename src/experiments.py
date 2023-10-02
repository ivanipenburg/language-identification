import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
# from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm

from data import LANGUAGE_CODES, get_dataloaders
from models import SimpleLSTM, SimpleTransformer
from stream import predict_streaming_batch


# def plot_confusion_matrix(model, dataloaders, config, languages):
    # """
    # Function that creates the confusion matrix
    # """

    # model.to(config['device'])
    # model.eval()
    # hidden = None

    # true_labels = []
    # predicted_labels = []

    # with torch.no_grad():
    #     for batch in tqdm(dataloaders['test']):
    #         inputs = batch['input_ids'].to(config['device'])
    #         labels = batch['label'].to(config['device'])

    #         logits, hidden = model(inputs.to(config['device']), hidden)
    #         prediction = torch.argmax(logits, dim=-1)

    #         true_labels.extend(labels.cpu().numpy())
    #         predicted_labels.extend(prediction.cpu().numpy())

    # # Compute the confusion matrix
    # confusion = confusion_matrix(true_labels, predicted_labels, labels=np.arange(len(languages)))

    # # Create a ConfusionMatrixDisplay
    # disp = ConfusionMatrixDisplay(confusion, display_labels=languages)

    # # Plot the confusion matrix
    # plt.figure(figsize=(10, 8))
    # disp.plot(cmap=plt.cm.Blues, values_format='d')
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()

# def performance_over_tokens(model, dataloaders, config, groups=None, int_to_label=None):
#     """
#     Function that plots the performance of the model over the number of tokens
#     """
#     confidence_per_token = []
#     correct_per_token = []
#     total_per_token = []

#     for batch in tqdm(dataloaders['test']):
#         predictions, labels, confidences = predict_streaming_batch(batch, model, config)

#         for i, prediction in enumerate(predictions[:64]):
#             if i >= len(correct_per_token):
#                 correct_per_token.append([])
#                 total_per_token.append([])
#                 confidence_per_token.append([])
#             if prediction == labels:
#                 correct_per_token[i].append(1)
#             else:
#                 correct_per_token[i].append(0)

#             total_per_token[i].append(1)
#             confidence_per_token[i].append(confidences[i][0][prediction])

#     accuracy_per_token = [np.sum(correct) / np.sum(total) for correct, total in zip(correct_per_token, total_per_token)]
#     confidence_per_token = [np.mean(confidence) for confidence in confidence_per_token]

#     plt.plot(accuracy_per_token)
#     plt.xlabel('Number of tokens')
#     plt.ylabel('Accuracy')
#     plt.show()

#     plt.plot(confidence_per_token)
#     plt.xlabel('Number of tokens')
#     plt.ylabel('Confidence')
#     plt.show()

def performance_over_tokens(model, dataloaders, config, groups=None, int_to_label=None):
    """
    Function that plots the performance of the model over the number of tokens for different groups.
    """
    if groups is None or int_to_label is None:
        raise ValueError("Both 'groups' and 'int_to_label' must be provided.")

    # Initialize data containers for each group
    group_data = {group_name: {'correct_per_token': [], 'total_per_token': [], 'confidence_per_token': []}
                  for group_name in groups.keys()}

    print(int_to_label)

    # Get all the unique value in dict of lists
    group_languages = set([language for languages in groups.values() for language in languages])
    print(group_languages)

    for batch in tqdm(dataloaders['test']):
        label = int_to_label[batch['label'].item()]
        if label not in group_languages:
            continue

        predictions, labels, confidences = predict_streaming_batch(batch, model, config)

        for i, prediction in enumerate(predictions[:64]):
            for group_name, group_labels in groups.items():
                if label in group_labels:
                    if i >= len(group_data[group_name]['correct_per_token']):
                        group_data[group_name]['correct_per_token'].append([])
                        group_data[group_name]['total_per_token'].append([])
                        group_data[group_name]['confidence_per_token'].append([])

                    if prediction == labels:
                        group_data[group_name]['correct_per_token'][i].append(1)
                    else:
                        group_data[group_name]['correct_per_token'][i].append(0)

                    group_data[group_name]['total_per_token'][i].append(1)
                    group_data[group_name]['confidence_per_token'][i].append(confidences[i][0][prediction])
                    
                    break

    # Plot performance for each group
    for group_name, data in group_data.items():
        accuracy_per_token = [np.sum(correct) / np.sum(total) for correct, total in zip(data['correct_per_token'], data['total_per_token'])]
        confidence_per_token = [np.mean(confidence) for confidence in data['confidence_per_token']]

        plt.plot(accuracy_per_token, label=group_name)

    plt.xlabel('Number of tokens')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.savefig('plots/accuracy_per_token.png')

    # Plot confidence for each group
    for group_name, data in group_data.items():
        confidence_per_token = [np.mean(confidence) for confidence in data['confidence_per_token']]

        plt.plot(confidence_per_token, label=group_name)

    plt.xlabel('Number of tokens')
    plt.ylabel('Confidence')
    plt.legend(loc='lower right')
    plt.savefig('plots/confidence_per_token.png')



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
        performance_over_tokens(model, dataloaders, config, LANGUAGE_GROUPS, int_to_label)
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