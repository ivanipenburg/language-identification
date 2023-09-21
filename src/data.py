import os

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

DATA_DIR = 'data'
DATASET_NAME = 'WiLI'

splits = {
    'train': 'x_train.txt',
    'test': 'x_test.txt'
}

def load_wili_dataset(data_dir):
    datasets = load_dataset('text', data_files={'train': 'data/x_train.txt', 'test': 'data/x_test.txt'})

    # Load the labels (y_train.txt, y_test.txt) and add them to the datasets
    for split in datasets:
        labels_filename = f'y_{split}.txt'

        with open(os.path.join(DATA_DIR, labels_filename), 'r', encoding='utf-8') as y_file:
            labels = y_file.read().splitlines()
        datasets[split] = datasets[split].add_column('label', labels)

    return datasets

def preprocess_datasets(datasets):
    # Tokenize and preprocess the data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    label_to_int = {label: i for i, label in enumerate(sorted(set(datasets['train']['label']).union(set(datasets['test']['label']))))}
    print(label_to_int)

    def tokenize(batch):
        tokenized_batch = tokenizer(batch['text'], padding='max_length', truncation=True)
        tokenized_batch['label'] = [label_to_int[label] for label in batch['label']]  
        
        return tokenized_batch

    print('Tokenizing datasets...')
    tokenized_datasets = {
        split: datasets[split].map(
            tokenize,
            batched=True,
            batch_size=512,
            remove_columns=['text'],
        )
        for split in datasets
    }

    print('Converting datasets to PyTorch tensors...')
    # Convert the datasets to PyTorch tensors
    tokenized_datasets = {
        split: tokenized_datasets[split].with_format('torch')
        for split in tokenized_datasets
    }

    for split in tokenized_datasets:
        tokenized_datasets[split].vocab_size = tokenizer.vocab_size

    return tokenized_datasets

def get_dataloaders(tokenize_datasets=True, dev_mode=False):
    # Load the dataset
    datasets = load_wili_dataset(DATA_DIR)

    # Test on only 4 languages
    if dev_mode:
        languages = ['eng', 'deu', 'fra', 'nld']
        for split in datasets:
            datasets[split] = datasets[split].filter(lambda example: example['label'] in languages)


    # Preprocess the dataset
    if tokenize_datasets:
        datasets = preprocess_datasets(datasets)

    # Create the dataloaders
    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=32,
            shuffle=True,
        )
        for split in datasets
    }

    return dataloaders
