import torch

from data import get_dataloaders
from models import SimpleLSTM, SimpleTransformer


def predict_streaming_batch(batch, model, config):
    model.eval()
    hidden = None
    predictions = []
    confidences = []
    with torch.no_grad():
        for input_id in batch['input_ids'][0]:
            # If input_id is 30000, it is a padding token
            if input_id == 30000:
                break

            input_id = input_id.unsqueeze(0).unsqueeze(0)
            labels = batch['label'].to(config['device'])
            logits, hidden = model(input_id.to(config['device']), hidden)
            prediction = torch.argmax(logits, dim=-1)
            confidence = torch.softmax(logits, dim=-1)
            predictions.append(prediction)
            confidences.append(confidence)
    return predictions, labels, confidences



def predict_streaming(dataloaders, model):
    for batch in dataloaders['test']:
        predictions, labels, confidences = predict_streaming_batch(batch, model)
        print(predictions)
        print(confidences)
        print(labels)
        break
    
if __name__ == '__main__':
    config = {
        'model': 'lstm',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'num_languages': 235,
        'hidden_dim': 128 ,
        'embedding_dim': 100,
        'model_checkpoint': 'checkpoints/model_4.pt'
    }

    dev_mode = True

    dataloaders, _ = get_dataloaders(tokenize_datasets=True, dev_mode=dev_mode)

    if config['model'] == 'lstm':        
        model = SimpleLSTM(config)
    elif config['model'] == 'transformer':
        model = SimpleTransformer(config)
    else:
        raise NotImplementedError('Model not implemented')
    
    model.load_state_dict(torch.load(config['model_checkpoint']))
    model.to(config['device'])

    predict_streaming(dataloaders, model)


