import requests


def get_word_embedding(word):
    # Build the API query URL
    query_url = f"http://api.conceptnet.io/numberbatch/text?node=/c/en/{word}"
    # Send the request and get the response
    response = requests.get(query_url).json()
    # Extract the embedding from the response
    embedding = response["vector"]
    # Return the embedding as a list of floats
    return [float(x) for x in embedding]


# Test the function
print(get_word_embedding("dog"))  # should print a list of floats
print(get_word_embedding("cat"))  # should print a list of floats
print(get_word_embedding("apple"))  # should print a list of floats
