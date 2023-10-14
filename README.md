# Requirements
Tested on Python 3.11. Create an environment and install the requirements with:
```bash
pip install -r requirements.txt
```

# Data
Download the (https://zenodo.org/record/841984/files/wili-2018.zip?download=1)[dataset] and extract it into a folder called `data` in the root of the project.

# Model checkpoints
Please send a request to the author to obtain the model checkpoints.

# Training
To train the LSTM and transformer models, run the following commands:
```bash
python train.py --model [lstm|transformer] 
```

To train the N-gram model, run the following command:
```bash
python ngram_classifier.py
```

# Experiments
To run the experiments, run the following command:
```bash
python experiments.py --model [lstm|transformer] --experiment [experiment_name]
```