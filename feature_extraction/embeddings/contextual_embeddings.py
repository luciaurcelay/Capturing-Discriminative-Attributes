import torch
from transformers import BertModel, BertTokenizer
import numpy as np


def get_embeddings(sequence, tokenizer, model):

    # Tokenize sequence
    tokens = tokenizer.encode(sequence, add_special_tokens=True)

    # Convert the tokens to a tensor
    tokens_tensor = torch.tensor([tokens])
    
    # Extract the embeddings
    outputs = model(tokens_tensor)
    hidden_states = outputs[2][1:]
    token_embeddings = hidden_states[-1]
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]
    # BERT embeddings have 768 dimensions (change slice below)
    bert_embeddings = list_token_embeddings[0][:5]
    bert_embeddings = [round(num, 4) for num in bert_embeddings]

    return bert_embeddings



def generate_bert_embeddings(dataframe):

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load the BERT model
    model = BertModel.from_pretrained("bert-base-uncased",
           output_hidden_states = True)

    # Initialize column
    dataframe["bert_embedding"] = np.zeros

    for index, row in  dataframe[["word1", "word2", "pivot"]].iterrows():

        # Create an empty list to store the sentences which do not have embeddings
        false_rows = []

        # Extract the three words from the row
        word1 = row["word1"]
        word2 = row["word2"]
        word3 = row["pivot"]

        # Get BERT embeddings
        try:
            bert_embeddings = get_embeddings([word1, word2, word3], tokenizer, model)
            dataframe["bert_embedding"].iloc[index] = bert_embeddings
        
        except:
            dataframe["bert_embedding"].iloc[index] = np.nan
            false_rows.append(index)
            print("Sequence without BERT embedding")

    # delete rows which have not have embeddings
    if len(false_rows) > 0:
        dataframe = dataframe.drop(false_rows).reset_index()

    return dataframe