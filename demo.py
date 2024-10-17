import json

# Sample JSON data
data = '''{
  "user": "JohnDoe",
  "age": 30,
  "hobbies": ["reading", "coding", "gaming"],
  "address": {
    "street": "123 Main St",
    "city": "Sampletown",
    "zipcode": "12345"
  }
}'''

# Parse the JSON data
json_data = json.loads(data)

# Recursively tokenize JSON objects
def tokenize_json(obj):
    """Tokenizes a JSON object into a list of tokens.

    Args:
        obj: The JSON object to tokenize.

    Returns:
        A list of tokens representing the JSON object.
    """

    tokens = []  # Initialize an empty list to store the tokens

    if isinstance(obj, dict):  # If the object is a dictionary
        for key, value in obj.items():  # Iterate over key-value pairs
            tokens.append(f'KEY: {key}')  # Add the key as a token
            tokens.extend(tokenize_json(value))  # Recursively tokenize the value
    elif isinstance(obj, list):  # If the object is a list
        for item in obj:  # Iterate over items
            tokens.extend(tokenize_json(item))  # Recursively tokenize each item
    else:  # If the object is a scalar value
        tokens.append(f'VALUE: {str(obj)}')  # Add the value as a token

    return tokens  # Return the list of tokens

# Tokenize the parsed JSON
tokens = tokenize_json(json_data)

# Output the tokens
for token in tokens:
    print(token)
