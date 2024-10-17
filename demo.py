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



import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

# Vocabulary
vocab = {"winter": 0, "summer": 1, "hot": 2, "cold": 3, "snow": 4, "sun": 5}

# Convert words to tensor
def words_to_tensor(words):
    return torch.tensor([vocab[word] for word in words])

# Example dataset class
class CBOWDataset(Dataset):
    def __init__(self, context_data, target_data):
        self.context_data = [words_to_tensor(context) for context in context_data]
        self.target_data = [vocab[target] for target in target_data]

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, idx):
        return self.context_data[idx], self.target_data[idx]

# Example CBOW model
class CBOWModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        embedded = self.embeddings(context).mean(dim=1)  # Average context embeddings
        output = self.linear(embedded)  # No need to reshape embedded
        return output

    def training_step(self, batch, batch_idx):
        context, target = batch
        output = self(context)
        loss = torch.nn.functional.cross_entropy(output, target.view(-1))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

if __name__ == "__main__":
    # Sample training data
    context_data = [["winter", "snow"], ["summer", "sun"], ["hot", "summer"], ["cold", "winter"], ["sun", "hot"], ["snow", "cold"]]
    target_data = ["cold", "hot", "sun", "snow", "summer", "winter"]

    train_dataset = CBOWDataset(context_data, target_data)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = CBOWModel(vocab_size=len(vocab), embedding_dim=10)

    # Train the model
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, train_loader)

    # Print embeddings for words in the context
    for word, index in vocab.items():
        word_tensor = torch.tensor(index).long()
        word_embed = model.embeddings(word_tensor)
        print(f"Embedding for '{word}': {word_embed.detach().numpy()}")

    # Example usage for prediction
    CTX = ["winter", "snow"]
    context = words_to_tensor(CTX)  # Context: ["winter", "snow"]
    print(CTX)
    model.eval()
    with torch.no_grad():
        output = model(context.view(1, -1))
        predicted_index = torch.argmax(output)
        print(f"Predicted next word: {list(vocab.keys())[list(vocab.values()).index(predicted_index.item())]}")
