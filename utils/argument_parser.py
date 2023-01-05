import argparse


def parse_input_arguments():

    parser = argparse.ArgumentParser(
        prog="Capturing Discriminative Attributes",
        description="Train different classifiers con Capturing Discriminative Attributes dataset",
    )

    parser.add_argument(
        "-r",
        "--relations",
        help="ConceptNet relationships",
        choices=["True", "False"],
        default="False",
    )
    parser.add_argument(
        "-e",
        "--embedding",
        help="Embedding type",
        choices=["glove", "contextual"],
        default="glove",
    )
    parser.add_argument(
        "-ge",
        "--glove_embedding",
        help="Glove pre-trained word embedding",
        choices=["glove.6B.50d", "glove.6B.100d", "glove.6B.200d", "glove.6B.300d"],
        default=["glove.6B.50d"],
    )
    parser.add_argument(
        "-ed",
        "--embedding_dim",
        help="Dimension of the embedding",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-nw",
        "--new_experiment",
        help="Create new experiment folder",
        choices=["True", "False"],
        default="True",
    )
    parser.add_argument("-s", "--seed", help="Seed", type=int, default=42)
    parser.add_argument(
        "-m",
        "--model",
        help="Trainer to model the data",
        choices=["lazy", "SVC", "XGBoost", "CNN"],
        default="SVC",
    )
    parser.add_argument(
        "-k",
        "--kernel",
        help="SVC kernel parameter",
        choices=["linear", "poly", "rbf", "sigmoid", "precomputed"],
        default="rbf",
    )
    parser.add_argument("-c", "--c", help="SVC C parameter", type=float, default=1.0)
    parser.add_argument(
        "-g",
        "--gamma",
        help="SVC gamma parameter",
        choices=["scale", "auto"],
        default="scale",
    )
    parser.add_argument(
        "-l1",
        "--l1_norm",
        help="Compute L1 norm",
        choices=["True", "False"],
        default="True",
    )
    parser.add_argument(
        "-cos",
        "--cosine",
        help="Compute cosine similarity",
        choices=["True", "False"],
        default="True",
    )

    parsed_args = parser.parse_args()
    return parsed_args
