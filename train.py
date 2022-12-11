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
from feature_extraction.embeddings import generate_embeddings

# Define Functions
def extract_features(parsed_args, train_df, val_df, feature_names, embeddings_path):
    start = time.time()
    print('EXTRACTING FEATURES FROM TRAINING SET')
    train_df = generate_embeddings(parsed_args, train_df, feature_names, embeddings_path)
    print('EXTRACTING FEATURES FROM VALIDATION SET')
    val_df = generate_embeddings(parsed_args, val_df, feature_names, embeddings_path)
    end = round(time.time()-start)
    print(f'This process took {end} seconds')
    return train_df, val_df

def train(new_exp_path, parsed_args, train_df, val_df):
    # prepare data folds
    X_train = train_df.drop(['word1', 'word2', 'pivot', 'label'], axis=1)
    y_train = train_df['label']
    X_val = val_df.drop(['index','word1', 'word2', 'pivot', 'label'], axis=1)
    y_val = val_df['label']
    # print    
    print('X_train dataframe')
    print(X_train.head(5))
    print('X_val dataframe')
    print(X_val.head(5))
    # select model
    selected_model = parsed_args.model
    # support vector classifier
    if selected_model == 'lazy':
        # clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
        # models,predictions = clf.fit(X_train, X_test, y_train, y_test)
        # print(models)
        pass
    elif selected_model == 'SVC':
        model = SVC_Trainer(seed, parsed_args.kernel, parsed_args.c, parsed_args.gamma)
        print('TRAINING CLASSIFIER')
        start = time.time()
        predictions = model.train_classifier(X_train, y_train, X_val)
        end = round(time.time()-start)
        print(f'This process took {end} seconds')
        # TBD model.save_model(new_exp_path)
        # TBD model.save_features(new_exp_path)
        val_accuracy = accuracy_score(y_val, predictions)
        print(f'Validation accuracy: {val_accuracy}')
        classification_rep = classification_report(y_val, predictions)
        print(classification_rep)
    # XGBoost
    elif selected_model == 'XGBoost':
        pass
    # convolutional neural network
    elif selected_model == 'CNN':
        pass


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
    seed = parsed_args.seed

    # Create train and validation splits
    train_df, val_df = create_train_val_dataframes(train_csv_path, validation_csv_path)
    test_df = create_test_dataframe(test_csv_path)
    """ print(f'Train dataframe:\n {train_df.head()}')
    print(f'Validation dataframe:\n {val_df.head()}')
    print(f'Test dataframe:\n {test_df.head()}') """
    
    # Extract features
    train_df_feat, val_df_feat = extract_features(parsed_args, train_df, val_df, feature_names, embeddings_path)

    # Train
    train(new_experiment_path, parsed_args, train_df_feat, val_df_feat)
    
    pass