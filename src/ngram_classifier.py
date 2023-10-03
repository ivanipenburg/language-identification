import torch

from tqdm import tqdm
from data import get_dataloaders
from sklearn.naive_bayes import MultinomialNB

if __name__ == '__main__':
    multinomial_nb_classifier = MultinomialNB()
    dataset = get_dataloaders(dev_mode=True)
    vocab_size = dataset['train'].dataset.vocab_size
    classes = set()

    for batch in dataset['train']:
        classes.update(batch['label'].tolist())

    for batch in tqdm(dataset['train'], desc='Fitting Multinomial Naive Bayes classiier'):
        # we need to keep track of the ngrams in a batch; since we're working with batches
        # we cannot make one giant ngram matrix -> we're partially fitting a classifier based on 
        # the counts of tokens in a given batch
        ngram_matrix = torch.zeros((len(batch['input_ids']), vocab_size))

        # count all occurrences of tokens in batch to create ngram matrix
        ngram_matrix.scatter_add_(1, batch['input_ids'], torch.ones(ngram_matrix.shape))
        # remove the first column which holds the number of padding tokens
        ngram_matrix = ngram_matrix[:, 1:]

        # partially fit the batch on the classifier
        multinomial_nb_classifier.partial_fit(ngram_matrix, batch['label'], list(classes))
    
    accuracies = []
    
    for batch in tqdm(dataset['test'], desc='Examining accuracy of classifier on test set'):
        ngram_matrix = torch.zeros((len(batch['input_ids']), vocab_size))
        ngram_matrix.scatter_add_(1, batch['input_ids'], torch.ones(ngram_matrix.shape))
        ngram_matrix = ngram_matrix[:, 1:]

        predictions = multinomial_nb_classifier.predict(ngram_matrix)
        correct_predictions = torch.sum(torch.Tensor(predictions) == batch['label']).item()
        total = len(batch['label'])
        batch_accuracy = correct_predictions / total
        accuracies.append(batch_accuracy)
    
    final_accuracy = torch.mean(torch.Tensor(accuracies)).item()
    print(f'The accuracy of the naive bayes n-gram classifier is: {final_accuracy}')