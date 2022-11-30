# Import Packages
import tensorflow as tf

# Import Functions
from utils.argument_parser import parse_input_arguments
from utils.path_utils import *
from utils.csv_utils import *
from feature_extraction.embeddings import generate_embeddings

# Define Functions
def extract_features(parsed_args, train_df, val_df, feature_names, embeddings_path):
    train_df = generate_embeddings(parsed_args, train_df, feature_names, embeddings_path)
    val_df = generate_embeddings(parsed_args, val_df, feature_names, embeddings_path)
    return train_df, val_df

def train(new_exp_path, parsed_args, train_df, val_df):
    return None



# Define main block
if __name__ == '__main__':

    # Parse arguments
    parsed_args = parse_input_arguments()
    
    # Set paths
    root_path = './'
    data_path = join_path(root_path, 'data')
    train_csv_path = join_path(root_path, 'data', 'training', 'train.csv')
    validation_csv_path = join_path(root_path, 'data', 'training', 'validation.csv')
    test_csv_path = join_path(root_path, 'data', 'test', 'ref', 'test.csv')
    experiments_paths = create_folder(root_path, 'experiments')
    new_experiment_path = create_new_experiment_folder(parsed_args, experiments_paths)
    embeddings_path = join_path(root_path, 'resources\glove.6B')
    feature_names = ['word1', 'word2', 'pivot', 'label']

    # Generate random seeds
    tf.random.set_seed(42)
    tf.keras.backend.clear_session()

    # Create train and validation splits
    train_df, val_df = create_train_val_dataframes(train_csv_path, validation_csv_path)
    test_df = create_test_dataframe(test_csv_path)
    """ print(f'Train dataframe:\n {train_df.head()}')
    print(f'Validation dataframe:\n {val_df.head()}')
    print(f'Test dataframe:\n {test_df.head()}') """
    
    # Extract features
    train_df_feat, val_df_feat = extract_features(parsed_args, train_df, val_df, feature_names, embeddings_path)
    
    # Train

    
    pass