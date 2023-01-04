import requests

RELATIONS_STANDARD = [

  "RelatedTo",
  "IsA",
  "UsedFor",
  "HasContext"

]

RELATIONS_SWAPPED = [

  "PartOf",
  "CapableOf"

]

ALL_RELATIONS = RELATIONS_STANDARD + RELATIONS_SWAPPED


# This function get the relationships between two word pairs
def get_relations(word1, word2, relations_standard, relations_swapped):

  # Build the API query URL
  query_url = f'http://api.conceptnet.io/query?node1=/c/en/{word1}&node2=/c/en/{word2}&language=en'

  # Send the request and get the response
  response = requests.get(query_url).json()

  # Extract the list of edges from the response
  edges = response['edges']

  # If the list is empty, there are no English language relationships between the words
  if len(edges) == 0:
    return []

  # Otherwise, extract the relevant English language relationships from the edges
  ## For standard order relations (word, attribute)
  relationships_standard = [edge['rel']['label'] for edge in edges if edge['rel']['label'] in relations_standard]
  unique_list = set(relationships_standard)
  relationships_standard = list(unique_list)

  ## For swapped order relations (attribute, word)
  relationships_swapped = [edge['rel']['label'] for edge in edges if edge['rel']['label'] in relations_swapped]
  unique_list = set(relationships_swapped)
  relationships_swapped = list(unique_list)

  # Return the list of relationships
  return relationships_standard + relationships_swapped


# Create new columns in dataset
def prepare_dataframe(dataframe):

  column_names = ['RelatedTo_att1', 'RelatedTo_att2', 'IsA_att1', 'IsA_att2',
  'UsedFor_att1', 'UsedFor_att2', 'HasContext_att1', 'HasContext_att2',
  'PartOf_att1', 'PartOf_att2', 'CapableOf_att1', 'CapableOf_att2']
  
  # Create a dictionary of column names and default values
  columns = {column_name: 0 for column_name in column_names}

  # Add the new columns to the dataframe
  dataframe = dataframe.assign(**columns)

  return dataframe


# Main function from the file
def extract_relations(dataframe):

  # CHANGE RANGE
  dataframe = dataframe.iloc[0:500]

  # Prepare dataframe creating new columns
  dataframe = prepare_dataframe(dataframe)

  for index, value in dataframe["pivot"].iteritems():

    print(f'Current row: {index}')

    attribute = value
    word1 = dataframe['word1'][index]
    word2 = dataframe['word2'][index]

    # Extract relations for each pair of word-attribute
    rel_1 = get_relations(word1, attribute, RELATIONS_STANDARD, RELATIONS_SWAPPED)
    rel_2 = get_relations(word2, attribute, RELATIONS_STANDARD, RELATIONS_SWAPPED)

    # Add 1 in the position that relation has been found
    for rel in rel_1:
      dataframe[rel+'_att1'][index] = 1

    for rel in rel_2:
      dataframe[rel+'_att2'][index] = 1

  return dataframe