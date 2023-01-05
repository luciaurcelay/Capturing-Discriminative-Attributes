# Capturing-Discriminative-Attributes

This is a project based on shared task [SemEval 2018 Task 10](https://aclanthology.org/S18-1117/)

## To-Do List
### General
- [x] Define folder and file structure
- [x] Implement main function and helpers
### Embeddings
- [ ] Fix GloVe embeddings (all dimension from the word embedding vectors should be used, but rn they are averaged to one dimension bc it yields an error if not)
- [ ] Implement contextual embedding extraction function
- [ ] ~~Implement ConceptNet embedding extraction function
- [ ] Implement word2vec and FastText (?)
### Knowledge Base
- [x] Study Luminoso's paper to see which kind of relationships are meaninful to be extracted
- [x] Implement relationship extraction function
- [ ] Generate .csv with the relationships (in progress)
## Other features
- [x] L1 Norm between word1-word2, word1-attribute, word2-attribute
- [x] Cosine similarity between word1-word2, word1-attribute, word2-attribute
### Classifiers
- [ ] Implement XGBoost
- [ ] Implement Convolutional Neural Network (and plain MLP)
### Experiments
- [ ] Make experiments for all configurations (ablation study)
### Project presentation
- [ ] Prepare slides

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
