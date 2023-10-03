import os

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

DATA_DIR = 'data'
DATASET_NAME = 'WiLI'
TOKENIZER_FILE = 'BPE_trained.json'

LANGUAGE_CODES = [
    'ace', 'afr', 'als', 'amh', 'ang', 'ara', 'arg', 'arz', 'asm', 'ast',
    'ava', 'aym', 'azb', 'aze', 'bak', 'bar', 'bcl', 'be-tarask', 'bel', 'ben',
    'bho', 'bjn', 'bod', 'bos', 'bpy', 'bre', 'bul', 'bxr', 'cat', 'cbk',
    'cdo', 'ceb', 'ces', 'che', 'chr', 'chv', 'ckb', 'cor', 'cos', 'crh',
    'csb', 'cym', 'dan', 'deu', 'diq', 'div', 'dsb', 'dty', 'egl', 'ell',
    'eng', 'epo', 'est', 'eus', 'ext', 'fao', 'fas', 'fin', 'fra', 'frp',
    'fry', 'fur', 'gag', 'gla', 'gle', 'glg', 'glk', 'glv', 'grn', 'guj',
    'hak', 'hat', 'hau', 'hbs', 'heb', 'hif', 'hin', 'hrv', 'hsb', 'hun',
    'hye', 'ibo', 'ido', 'ile', 'ilo', 'ina', 'ind', 'isl', 'ita', 'jam',
    'jav', 'jbo', 'jpn', 'kaa', 'kab', 'kan', 'kat', 'kaz', 'kbd', 'khm',
    'kin', 'kir', 'koi', 'kok', 'kom', 'kor', 'krc', 'ksh', 'kur', 'lad',
    'lao', 'lat', 'lav', 'lez', 'lij', 'lim', 'lin', 'lit', 'lmo', 'lrc',
    'ltg', 'ltz', 'lug', 'lzh', 'mai', 'mal', 'map-bms', 'mar', 'mdf', 'mhr',
    'min', 'mkd', 'mlg', 'mlt', 'mon', 'mri', 'mrj', 'msa', 'mwl', 'mya',
    'myv', 'mzn', 'nan', 'nap', 'nav', 'nci', 'nds', 'nds-nl', 'nep', 'new',
    'nld', 'nno', 'nob', 'nrm', 'nso', 'oci', 'olo', 'ori', 'orm', 'oss',
    'pag', 'pam', 'pan', 'pap', 'pcd', 'pdc', 'pfl', 'pnb', 'pol', 'por',
    'pus', 'que', 'roa-tara', 'roh', 'ron', 'rue', 'rup', 'rus', 'sah',
    'san', 'scn', 'sco', 'sgs', 'sin', 'slk', 'slv', 'sme', 'sna', 'snd',
    'som', 'spa', 'sqi', 'srd', 'srn', 'srp', 'stq', 'sun', 'swa', 'swe',
    'szl', 'tam', 'tat', 'tcy', 'tel', 'tet', 'tgk', 'tgl', 'tha', 'ton',
    'tsn', 'tuk', 'tur', 'tyv', 'udm', 'uig', 'ukr', 'urd', 'uzb', 'vec',
    'vep', 'vie', 'vls', 'vol', 'vro', 'war', 'wln', 'wol', 'wuu', 'xho',
    'xmf', 'yid', 'yor', 'zea', 'zh-yue', 'zho'
]

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

def train_BPE(languages=None):

    # create a .txt file that only contains the specific languages on which we want to train the BPE tokenizer
    output_path = os.path.join(DATA_DIR, 'BPE_train.txt')
    with open(output_path, 'w', encoding='utf-8') as output_file:
        with open(os.path.join(DATA_DIR,'x_train.txt' ), 'r', encoding='utf-8') as x_file, open(os.path.join(DATA_DIR,'y_train.txt' ), 'r', encoding='utf-8') as y_file:
            for x1, y1 in zip(x_file, y_file):
                x1 = x1.strip()
                y1 = y1.strip()
                if languages is None or y1 in languages:
                    output_file.write(x1 + '\n')

    unk_token = "<UNK>"  # token for unknown words
    spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]
    file = [output_path]

    tokenizer = Tokenizer(BPE(unk_token = unk_token))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens = spl_tokens)
    tokenizer.train(file, trainer)
    tokenizer.save(TOKENIZER_FILE)

def preprocess_datasets(datasets, train_BPE_flag, languages=None):

    if train_BPE_flag:
       train_BPE(languages)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE, model_max_length=512)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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

    return tokenized_datasets

def get_dataloaders(tokenize_datasets=True, dev_mode=False, train_BPE=True, batch_size=32):
    """
    Function that loads the dataloaders

    Args:
    - tokenize_datasets (Boolean): whether or not to pre-process the dataset
    - dev_mode (Boolean):
    - trained_BPE: should contain the datapath to a pre_trained BPE (string) or else
    will have the Boolean value False
    """


    # Load the dataset
    datasets = load_wili_dataset(DATA_DIR)
    languages = None

    # Test on only 4 languages
    if dev_mode:
        languages = ['eng', 'deu', 'fra', 'nld']
        for split in datasets:
            datasets[split] = datasets[split].filter(lambda example: example['label'] in languages)


    # Preprocess the dataset
    if tokenize_datasets:
        datasets = preprocess_datasets(datasets, train_BPE, languages)

    # Create the dataloaders
    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=True,
        )
        for split in datasets
    }

    return dataloaders
