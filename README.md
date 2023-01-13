# Capturing-Discriminative-Attributes

This is a project based on shared task [SemEval 2018 Task 10](https://aclanthology.org/S18-1117/)

## Installation
### Clone repository

`git clone https://github.com/luciaurcelay/Capturing-Discriminative-Attributes.git`

### Install requirements
`pip install -r requirements.txt`

### Download required resources
* GloVe embeddings
`https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip`

Once downloaded, unzip in `resources` folder. Files should follow the following structure:
    
    resources\glove.6B\glove.6B.50d.txt
    resources\glove.6B\glove.6B.100d.txt
    resources\glove.6B\glove.6B.200d.txt
    resources\glove.6B\glove.6B.300d.txt

## Usage
### Train classifier
`train.py` function requires argurments, use the following command to take a look at them:

`python train.py -h`
