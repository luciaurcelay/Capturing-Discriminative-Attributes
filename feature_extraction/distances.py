import math
from itertools import combinations

# Calculate the computation of l1 norm
def calculate_l1_norm(vec1, vec2):

    # Check that the vectors have the same length
    if len(vec1) != len(vec2):
        print("Vectors must have the same length")
        norm = 0

    # Calculate the L1 normelse:
    else:
        norm = sum(abs(a - b) for a, b in zip(vec1, vec2))

    return norm


def calculate_l1_norm_pandas(attr1, attr2):
    norm = abs(attr1.astype(float).values - attr2.astype(float).values).sum(axis=1)
    return norm


# Calculate the computation of cosine similarity
def calculate_cosine_similarity(vec1, vec2):

    # Check that the vectors have the same length
    if len(vec1) != len(vec2):
        print("Vectors must have the same length")
        similarity = 0
    else:
        # Calculate the dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Calculate the magnitudes of the vectors
        magnitude1 = math.sqrt(sum(a**2 for a in vec1))
        magnitude2 = math.sqrt(sum(b**2 for b in vec2))

        # Calculate the cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)

    return similarity


# This function computes: L1_word1_word2, L1_word1_attribute, L2_word2_attribute
def compute_l1_norm(dataframe):

    dataframe = dataframe.assign(l1_12=None, l1_13=None, l1_23=None)

    attr_combinations = combinations(["word1", "word2", "pivot"], 2)
    colnames = ["12", "13", "23"]

    for colname, (attr1, attr2) in zip(colnames, attr_combinations):
        vec1 = dataframe[[col for col in dataframe.columns if attr1 in col][1:]]
        vec2 = dataframe[[col for col in dataframe.columns if attr2 in col][1:]]

        dataframe[f"l1_{colname}"] = calculate_l1_norm_pandas(vec1, vec2)

    # Still keep this for possible debugging
    # for index, value in dataframe["pivot_embedding_dim_0"].iteritems():

    #     attribute = value
    #     word1 = dataframe["word1_embedding"][index]
    #     word2 = dataframe["word2_embedding"][index]

    #     d1_2 = calculate_l1_norm(word1, word2)
    #     d1_3 = calculate_l1_norm(word1, attribute)
    #     d2_3 = calculate_l1_norm(word2, attribute)

    #     dataframe["l1_12"][index] = d1_2
    #     dataframe["l1_13"][index] = d1_3
    #     dataframe["l1_23"][index] = d2_3

    return dataframe


def compute_cosine_similarity(dataframe):

    dataframe = dataframe.assign(cosine_12=None, cosine_13=None, cosine_23=None)

    for index, value in dataframe["pivot_embedding"].iteritems():

        attribute = value
        word1 = dataframe["word1_embedding"][index]
        word2 = dataframe["word2_embedding"][index]

        d1_2 = calculate_cosine_similarity(word1, word2)
        d1_3 = calculate_cosine_similarity(word1, attribute)
        d2_3 = calculate_cosine_similarity(word2, attribute)

        dataframe["cosine_12"][index] = d1_2
        dataframe["cosine_13"][index] = d1_3
        dataframe["cosine_23"][index] = d2_3

    return dataframe
