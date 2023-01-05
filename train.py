# Import Packages
import time
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from lazypredict.Supervised import LazyClassifier

# Import Functions
from utils.argument_parser import parse_input_arguments
from utils.path_utils import *
from utils.csv_utils import *
from models.svc import SVC_Trainer
from models.xgboost import XGBClassifier
from feature_extraction.ConceptNet.conceptnet import extract_relations
from feature_extraction.embeddings.glove_embeddings import generate_glove_embeddings
from feature_extraction.distances import compute_l1_norm, compute_cosine_similarity

# Main helper functions
def extract_features(
    parsed_args, train_df, val_df, feature_names, embeddings_path, conceptnet_path
):

    start = time.time()

    ## Extract features from knowledge base
    conceptnet = parsed_args.relations
    # GloVe Embeddings
    if conceptnet == "True":

        print("EXTRACTING CONCEPTNET RELATIONS FROM TRAINING SET")
        # Relations from train set
        train_df = extract_relations(train_df)
        # Generate .csv file and save to resources folder
        print("SAVING TRAINING RELATIONS")
        train_df.to_csv(
            conceptnet_path + "/train_conceptnet_relations_1000_3000.csv", index=False
        )

        # print('EXTRACTING CONCEPTNET RELATIONS FROM VALIDATION SET')
        # Relations from validation set
        # val_df = extract_relations(val_df)
        # print('SAVING VALIDATION RELATIONS')

        # val_df.to_csv(conceptnet_path+"/val_conceptnet_relations_0_500.csv", index=False)

    else:
        pass

    ## Extract word embeddings from the attributes
    selected_embedding = parsed_args.embedding
    # GloVe Embeddings
    if selected_embedding == "glove":

        print("EXTRACTING FEATURES FROM TRAINING SET")
        train_df = generate_glove_embeddings(
            parsed_args, train_df, feature_names, embeddings_path
        )

        print("EXTRACTING FEATURES FROM VALIDATION SET")
        val_df = generate_glove_embeddings(
            parsed_args, val_df, feature_names, embeddings_path
        )

    # Contextual Emeddings
    elif selected_embedding == "contextual":

        print("EXTRACTING CONTEXTUAL EMBEDDINGS FROM TRAINING SET")
        # TODO

        print("EXTRACTING CONTEXTUAL FROM VALIDATION SET")
        # TODO
    
    
    ## Compute distances
    # L1 norm
    selected_l1 = parsed_args.l1_norm
    if selected_l1 == "True":
        train_df = compute_l1_norm(train_df)
        val_df = compute_l1_norm(val_df)
    else:
        pass

    # Cosine similarity
    selected_cosine = parsed_args.cosine
    if selected_cosine == "True":
        train_df = compute_cosine_similarity(train_df)
        val_df = compute_cosine_similarity(val_df)
    else:
        pass

    end = round(time.time() - start)
    print(f"Feature extraction took {end} seconds")

    return train_df, val_df


def train(new_exp_path, parsed_args, train_df, val_df):

    # prepare data folds
    X_train = train_df.drop(["word1", "word2", "pivot", "label"], axis=1)
    y_train = train_df["label"]
    X_val = val_df.drop(["index", "word1", "word2", "pivot", "label"], axis=1)
    y_val = val_df["label"]

    # print dataframes
    print("X_train dataframe")
    print(X_train.head(5))
    print("X_val dataframe")
    print(X_val.head(5))

    # Select model
    selected_model = parsed_args.model

    # Lazy classifier (tests different kinds of simple classifiers)
    if selected_model == "lazy":

        # clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
        # models,predictions = clf.fit(X_train, X_test, y_train, y_test)
        # print(models)

        pass

    # Support vector classifier
    elif selected_model == "SVC":

        # Import classifier
        model = SVC_Trainer(seed, parsed_args.kernel, parsed_args.c, parsed_args.gamma)

        # Search for the best parameters
        # best_params = model.search_best_params(X_train, y_train)
        # print(best_params)

    # XGBoost TODO
    elif selected_model == "XGBoost":

        # Import classifier
        model = XGBClassifier()

    # CNN (Convolutional Neural Network) TODO
    elif selected_model == "CNN":

        pass

    print("TRAINING CLASSIFIER")
    start = time.time()

    # Train the model
    predictions = model.train_classifier(X_train, y_train, X_val)
    end = round(time.time() - start)
    print(f"Training took {end} seconds")

    # TBD model.save_model(new_exp_path)
    # TBD model.save_features(new_exp_path)
    val_accuracy = accuracy_score(y_val, predictions)
    print(f"Validation accuracy: {val_accuracy}")

    # Make classification report
    classification_rep = classification_report(y_val, predictions)
    print(classification_rep)


# Define main block
if __name__ == "__main__":

    # Parse arguments
    parsed_args = parse_input_arguments()

    # Set paths
    root_path = "./"
    data_path = join_path(root_path, "data")
    train_csv_path = join_path(root_path, "data", "training", "train.csv")
    validation_csv_path = join_path(root_path, "data", "training", "validation.csv")
    test_csv_path = join_path(root_path, "data", "test", "ref", "test.csv")
    experiments_paths = create_folder(root_path, "experiments")
    new_experiment_path = create_new_experiment_folder(parsed_args, experiments_paths)
    embeddings_path = join_path(root_path, 'resources\glove.6B')
    conceptnet_path = join_path(root_path, 'resources\conceptnet_relations')
    feature_names = ['word1', 'word2', 'pivot', 'label']

    # Generate random seeds
    tf.random.set_seed(42)
    tf.keras.backend.clear_session()
    seed = parsed_args.seed

    # Create train and validation splits
    train_df, val_df = create_train_val_dataframes(train_csv_path, validation_csv_path)
    test_df = create_test_dataframe(test_csv_path)
    """ print(f'Train dataframe:\n {train_df.head()}')
    print(f'Validation dataframe:\n {val_df.head()}')
    print(f'Test dataframe:\n {test_df.head()}') """

    print(f"Train dataframe:\n {train_df.head()}")
    print(f"Validation dataframe:\n {val_df.head()}")
    print(f"Test dataframe:\n {test_df.head()}")

    # Extract features
    train_df_feat, val_df_feat = extract_features(
        parsed_args, train_df, val_df, feature_names, embeddings_path, conceptnet_path
    )

    # Train the selected classifier
    train(new_experiment_path, parsed_args, train_df_feat, val_df_feat)

    # Evaluation on test set. TODO

    pass
