import requests

def get_relationships(word1, word2):
  # Build the API query URL
  query_url = f'http://api.conceptnet.io/query?node1=/c/en/{word1}&node2=/c/en/{word2}'
  # Send the request and get the response
  response = requests.get(query_url).json()
  # Extract the list of edges from the response
  edges = response['edges']
  # If the list is empty, there are no relationships between the words
  if len(edges) == 0:
    return []
  # Otherwise, extract the relationships from the edges
  relationships = [edge['rel']['label'] for edge in edges]
  # Return the list of relationships
  return relationships

# Test the function
print(get_relationships('dog', 'cat')) # should print ['SimilarTo', 'IsA']
print(get_relationships('apple', 'orange')) # should print ['SimilarTo', 'IsA']
print(get_relationships('car', 'boat')) # should print ['IsA', 'HasProperty']