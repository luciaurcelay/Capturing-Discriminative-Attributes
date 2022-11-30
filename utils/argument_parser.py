import argparse

def parse_input_arguments():
    parser = argparse.ArgumentParser(
        prog='Capturing Discriminative Attributes', 
        description='Train different classifiers con Capturing Discriminative Attributes dataset')

    parser.add_argument('-e','--embedding', help='Pre-trained word embedding', choices=['glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d'], default=['glove.6B.50d'])
    parser.add_argument('-ed', '--embedding_dim', help='Dimension of the embedding', type=int, default=50)

    parsed_args = parser.parse_args()
    return parsed_args