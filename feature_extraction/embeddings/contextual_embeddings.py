import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


def get_embeddings(sequence, tokenizer, model, used_dim):

    # Tokenize sequence
    tokens = tokenizer.encode(list(sequence), add_special_tokens=True)

    # Convert the tokens to a tensor
    tokens_tensor = torch.tensor([tokens])

    # Extract the embeddings
    outputs = model(tokens_tensor, output_attentions=False)
    token_embeddings = outputs.hidden_states[-1].squeeze()[1:4, :used_dim]
    bert_embeddings = token_embeddings.ravel().detach().numpy()

    return bert_embeddings


def generate_bert_embeddings(dataframe, embed_dim):

    dataframe = dataframe[0:]
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load the BERT model
    model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

    # Init column names
    bert_colnames = []
    for word in ["word1", "word2", "pivot"]:
        for i in range(embed_dim):
            bert_colnames.append(f"{word}_bert_embedding_dim_{i+1}")

    # Get data
    data = []
    for index, row in tqdm(dataframe[["word1", "word2", "pivot"]].iterrows()):
        bert_embeddings = get_embeddings(row, tokenizer, model, embed_dim)
        data.append(bert_embeddings)

    dataframe = pd.concat([dataframe, pd.DataFrame(data, columns=bert_colnames)], axis=1)

    return dataframe
