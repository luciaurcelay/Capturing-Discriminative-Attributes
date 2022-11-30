import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict
from utils.path_utils import join_path


def generate_word_index(dataframe):
    word1_dict = dataframe['word1'].value_counts().to_dict()
    word2_dict = dataframe['word2'].value_counts().to_dict()
    pivot_dict = dataframe['pivot'].value_counts().to_dict()
    # concat dictionaries
    dicts = [word1_dict, word2_dict, pivot_dict]
    super_dict = defaultdict(set)
    # create dictionary merging words from all columns
    for d in dicts:
        for k, v in d.items():
            if not k in super_dict.keys():
                super_dict[k].add(v)
            else:
                super_dict[k].add(v)
    # sum the counts of the values of each key in dictionary
    for k, v in super_dict.items():
        list=[]
        for i in super_dict[k]:
            list.append(i)
        super_dict[k] = sum(list)
    return super_dict


def embedding_for_vocab(args, embeddings_path, word_index):
    embedding_dim = args.embedding_dim
    """ vocab_size = len(word_index) + 1

    embedding_matrix_vocab = np.zeros((vocab_size,
                                       embedding_dim)) """

    embedding_vocab_dict = {}
    
    with open(embeddings_path, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                """ idx = word_index[word]
                embedding_matrix_vocab[idx]= np.array(
                    vector, dtype=np.float32)[:embedding_dim] """
                
                embedding_vocab_dict[word] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
  
    return embedding_vocab_dict


def create_embeddings(dataframe, feature_names, embeddings_vocab):
    # initialize columns
    for i in range(len(feature_names)-1):
        dataframe[feature_names[i] + '_embedding'] = np.zeros
    # add embedding values to each word
    false_rows = []
    for (columnName, columnData) in dataframe[['word1', 'word2', 'pivot']].iteritems():
        for i, word in enumerate(columnData.values):
            try:
                dataframe[columnName + '_embedding'].iloc[i] = embeddings_vocab[word]
            except:
                dataframe[columnName + '_embedding'].iloc[i] = 'False'
                false_rows.append(i)
                print('False found')
    # delete rows which have rows that do not have embeddings
    if len(false_rows) > 0:
        dataframe.drop(dataframe.index[false_rows])
    
    return dataframe


def generate_embeddings(args, dataframe, feature_names, embeddings_path):
    # create word index
    word_index = generate_word_index(dataframe)
    # access embedding file
    embeddings_path = join_path(embeddings_path, f'{args.embedding}.txt')
    # create embedding vocabulary matrix
    embedding_matrix_vocab = embedding_for_vocab(args, embeddings_path, word_index)
    # create word embeddings for every word
    dataframe = create_embeddings(dataframe, feature_names, embedding_matrix_vocab)
    return dataframe