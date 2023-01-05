from path_utils import *
import pandas as pd

root_path = '../../'
csv_path = join_path(root_path, 'resources/conceptnet_relations')


file_dict = {

    '0_500': 'resources/conceptnet_relations/training_0_100.csv', 
    '500_1000': 'resources/conceptnet_relations/training_500_1000.csv',
    '1000_3000': 'resources/conceptnet_relations/training_1000_3000.csv'

    }


def merge_csv_files(file_dict):
    # Initialize an empty list to store the dataframes
    df_list = []
    
    # Iterate through the dictionary of CSV files
    for file_name, file_path in file_dict.items():
        # Read the CSV file into a dataframe
        df = pd.read_csv(file_path)
        
        # Add the dataframe to the list
        df_list.append(df)
        
    # Concatenate the dataframes into one
    merged_df = pd.concat(df_list)
    
    return merged_df


# Merge the CSV files
merged_df = merge_csv_files(file_dict)