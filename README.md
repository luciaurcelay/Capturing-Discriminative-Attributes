# Capturing-Discriminative-Attributes

This is a project based on shared task [SemEval 2018 Task 10](https://aclanthology.org/S18-1117/)

## To-Dos
### General
- [x] Define folder and file structure
- [x] Implement main function and helpers
### Embeddings
- [ ] Fix GloVe embeddings (all dimension from the word embedding vectors should be used, but rn they are averaged to one dimension bc it yields an error if not)
- [ ] Implement contextual embedding extraction function
- [ ] Implement ConceptNet embedding extraction function (there is a sample function in `feature_extraction\ConceptNet\conceptnet.py`)
### Knowledge Base
- [ ] Study Luminoso's paper to see which kind of relationships are meaninful to be extracted
- [ ] Implement relationship extraction function
### Classifiers
- [ ] Implement XGBoost
- [ ] Implement Convolutional Neural Network or other variation
### Experiments
- [ ] Make experiments for all configurations (ablation)
### Project presentation
- [ ] Prepare slides

## Run the code
### Clone repository
In order to run the code you must clone the repository using HTTPS:

`git clone https://github.com/luciaurcelay/Capturing-Discriminative-Attributes.git`

Or SSH:

`git clone git@github.com:luciaurcelay/Capturing-Discriminative-Attributes.git`

### Install requirements
`pip install -r requirements.txt`

### Execute code
train.py function requires argurments, use the following line get more information about them:

`python train.py -h`
