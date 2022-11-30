import pandas as pd
from utils.path_utils import join_path

def create_train_val_dataframes(train_csv_path, validation_csv_path):
    train_df = pd.read_csv(train_csv_path)
    # train_df.to_csv(join_path(data_path, 'training', 'train.csv'), index=None)
    validation_df = pd.read_csv(validation_csv_path)
    # validation_df.to_csv(join_path(data_path, 'training', 'validation.csv'), index=None)
    return train_df, validation_df

def create_test_dataframe(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    # test_df.to_csv(join_path(data_path, 'test', 'ref', 'test.csv'), index=None)
    return test_df