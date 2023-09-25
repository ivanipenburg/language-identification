import torch
import torch.nn as nn
from transformers import AutoTokenizer

from data import get_dataloaders
from models import SimpleLSTM


def predict_streaming_batch(batch, model):
    model.eval()
    hx = None
    predictions = []
    confidences = []
    with torch.no_grad():
        for input_id in batch['input_ids']:
            input_id = torch.tensor([input_id])
            logits, hx = model(input_id, hx)
            
            prediction = torch.argmax(logits, dim=-1)
            confidence = torch.softmax(logits, dim=-1)
            predictions.append(prediction)
            confidences.append(confidence)
    return predictions, confidences



def predict_streaming(dataloaders, model):
    for batch in dataloaders['test']:
        predictions, confidences = predict_streaming_batch(batch, model)
        print(predictions)
        print(confidences)
        break
    


if __name__ == '__main__':
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    model = SimpleBiLSTM()
    model.load_state_dict(torch.load('checkpoints/model.pt'))
    model.eval()

    model.to(config['device'])

    dataloaders = get_dataloaders(tokenize_datasets=True, dev_mode=True)


