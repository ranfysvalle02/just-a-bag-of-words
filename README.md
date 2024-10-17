# The Foundations of Text Representation

## How Early Text Representation Models Shaped Today’s AI

![](https://images.squarespace-cdn.com/content/v1/5f831c8e85a8ea3b5baabcbd/1615645049220-VSASS4Q65WTC8YON4FQ3/0*kBwh_sX7lsJxrOvr.png?format=2500w)

__Image Credit to [Word2Vec vs BERT](https://www.saltdatalabs.com/blog/word2vec-vs-bert)__

In the last few years, large language models have radically transformed how we interact with technology, from search engines to virtual assistants. Yet, behind LLM's incredible understanding of language lies a long history of evolving text representation techniques that have shaped the foundation of modern Natural Language Processing (NLP). This blog post takes a journey through the pioneering approaches that dominated before LLM’s emergence, exploring the humble beginnings of Bag of Words (BoW), the innovations of Word2Vec, and how these earlier methods helped us build the AI systems that today power some of the world's largest companies.

As artificial intelligence (AI) continues to integrate deeper into industries from healthcare to finance, understanding these historical methods isn't just academic—it's essential for grasping where NLP is heading in the future. 

## Tokenization: The First Step in Text Processing

Before we can even begin to represent text in a way that machines can understand, we need to break it down into smaller, manageable pieces. This is where tokenization comes in. Tokenization is the process of splitting a large paragraph or text into smaller units, typically words or phrases, called tokens. These tokens become the basic building blocks for further text analysis.

Tokenization might seem like a simple task, but it's more complex than just splitting text by spaces. Consider sentences with contractions like "don't" or "it's," or compound words like "mother-in-law." Different languages also have different rules for tokenization. For instance, in Chinese, words often consist of multiple characters with no spaces in between.

Here’s an illustration of tokenization using the popular Natural Language Toolkit (NLTK) in Python:

```python
import nltk

# Download necessary resources
nltk.download('punkt')

# Sample text
text = "Tokenization isn't as easy as it seems."

# Tokenize the text into words
tokens = nltk.word_tokenize(text)

print(tokens)
```

**Output:**
```
['Tokenization', 'is', "n't", 'as', 'easy', 'as', 'it', 'seems', '.']
```

### Key Insights:

- **Handling Contractions:** Notice how "isn't" is split into "is" and "n't", reflecting the underlying meaning more accurately.
- **Punctuation Handling:** The period at the end of the sentence is treated as a separate token, which can be crucial in certain NLP tasks like sentiment analysis or named entity recognition.

### **Why use Punkt?**

Punkt is designed to split sentences and words more intelligently than just splitting by spaces or punctuation marks. Here’s why it’s essential:

1. **Sentence and Word Tokenization**: Punkt handles both sentence and word tokenization, recognizing sentence boundaries and handling edge cases like abbreviations, which might confuse a simple space-based tokenizer.
   
2. **Language-Agnostic**: While primarily built for English, Punkt can be adapted for other languages with minimal modification.

3. **Handling Edge Cases**: Consider sentences like *"Dr. Smith isn't available."* A naive approach might split "Dr." as a standalone token, but Punkt is designed to recognize "Dr." as an abbreviation and not a sentence boundary.

4. **Contractions**: It handles contractions like "isn't" by splitting it into two tokens: "is" and "n't", which is important for many NLP tasks that require analyzing individual word units.

In short, Punkt helps make tokenization more accurate by handling linguistic nuances and complex tokenization scenarios that basic splitting functions might miss.

### Key points to remember about tokenization:

- It's the first step in text preprocessing for NLP tasks.
- It breaks down text into smaller units (tokens), which can be words, phrases, or even sentences.
- The choice of token depends on the task at hand. For instance, if you're analyzing sentiment, you might want to consider phrases (like "not good") instead of just words.
- Tokenization isn't always straightforward, especially for languages other than English. Specialized tokenizers might be needed for different languages or tasks.

## What about JSON?

When dealing with JSON (JavaScript Object Notation) data, tokenization becomes more nuanced. JSON is often used to represent structured data, containing nested objects, arrays, and key-value pairs. Simply splitting text by spaces won’t work here—you’ll need to preserve both the structure and the meaning of the data.

Let’s consider an example:

```json
{
  "user": "JohnDoe",
  "age": 30,
  "hobbies": ["reading", "coding", "gaming"],
  "address": {
    "street": "123 Main St",
    "city": "Sampletown",
    "zipcode": "12345"
  }
}
```

Tokenizing such an object means not just breaking it into individual words, but also handling keys, values, arrays, and nested objects while maintaining the structure. Here’s a simple Python example that uses the `json` module to tokenize and traverse this structure.

```python
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
```

**Explanation:**
- **Recursive Tokenization:** The function `tokenize_json` recursively traverses the JSON structure, identifying keys and values at every level, including nested objects and arrays.
- **Differentiating Between Keys and Values:** In the output, we can clearly see the distinction between JSON keys (like `"user"`, `"age"`, etc.) and their corresponding values.
- **Handling Nested Data:** This method ensures that even deeply nested structures like the `"address"` field are tokenized correctly.

**Output:**
```
KEY: user
VALUE: JohnDoe
KEY: age
VALUE: 30
KEY: hobbies
VALUE: reading
VALUE: coding
VALUE: gaming
KEY: address
KEY: street
VALUE: 123 Main St
KEY: city
VALUE: Sampletown
KEY: zipcode
VALUE: 12345
```

- **KEY tokens:** These represent the names of keys or properties within the JSON object. For example, in the JSON object `{"name": "Alice", "age": 30}`, the tokens would be `KEY: name` and `KEY: age`.
- **VALUE tokens:** These represent the values associated with the keys. In the same example, the VALUE tokens would be `VALUE: Alice` and `VALUE: 30`.

---

### Nuances of Tokenizing JSON:

1. **Preserving Structure:** One of the key challenges when tokenizing JSON data is maintaining its hierarchical structure. Simply flattening everything into a list of words (as you might with regular text) would lose the relationships between keys and values, which are crucial for many tasks.

2. **Handling Arrays:** In JSON, arrays can store multiple values, and tokenization needs to distinguish between each element within the array. For instance, the `"hobbies"` field is an array of strings, each representing a separate token, but we must ensure the array structure is respected.

3. **Keys vs. Values:** JSON consists of key-value pairs, so it’s important to clearly differentiate between keys (which act as labels) and values (which hold the actual data). When tokenizing, you may want to process keys differently from values, especially in tasks like data extraction or information retrieval.

4. **Nested Objects:** JSON often contains nested objects. For example, the `"address"` field is itself an object containing more key-value pairs. Tokenizing such nested structures requires a recursive approach to ensure the hierarchy is preserved.

---

#### Why Tokenizing JSON Matters:

Tokenizing JSON data is critical in real-world applications where textual data is structured (like web APIs, logs, or configuration files). Unlike free-form text, JSON has a specific format that must be maintained during tokenization. Proper tokenization ensures the machine can interpret both the structure and the content correctly, which is essential for tasks like:

- **Data extraction:** When pulling information from APIs or databases, maintaining the key-value relationships is crucial.
- **Entity recognition:** If you're building a model to recognize specific entities (like addresses or user information), you'll need to tokenize structured data accurately.
- **Preprocessing for Machine Learning:** Structured JSON data might feed into ML pipelines, and improper tokenization could cause models to misinterpret the input.

---

## [Learn More About JSON tokenization here ](https://github.com/ranfysvalle02/json-tokenization/)

Tokenization is a crucial step in text representation as it determines how the text will be broken down and understood by the subsequent models. Whether you're using Bag of Words, Word2Vec, or even more advanced models like GPT-4, tokenization is the first step in making text understandable to machines.

## Capturing Meaning in Text

Capturing the meaning in text is one of the most challenging aspects of text analysis. It involves understanding the context, semantics, and syntax of the text. Several techniques are used to capture meaning in text:

- **Part-of-Speech (POS) Tagging:** This involves labeling each word in the text as a noun, verb, adjective, etc. POS tagging helps in understanding the grammatical structure of the text and the role of each word in the sentence.

- **Named Entity Recognition (NER):** NER involves identifying and classifying named entities in the text, such as persons, organizations, locations, etc. It helps in understanding who or what the text is about.

- **Dependency Parsing:** This involves analyzing the grammatical structure of the sentence to determine the relationships between words. Dependency parsing helps in understanding how words in a sentence relate to each other.

- **Semantic Analysis:** This involves understanding the meaning of the text. Techniques used in semantic analysis include word sense disambiguation (determining the correct meaning of a word based on context), semantic role labeling (identifying the semantic roles of words in a sentence), and semantic similarity (measuring the semantic similarity between words or sentences).

## Word Embeddings: Capturing Semantic Relationships

Word embeddings are a type of word representation that allows words with similar meaning to have a similar representation. They are a distributed representation for text that is perhaps one of the key breakthroughs for the impressive performance of deep learning methods on challenging natural language processing problems.

Word embeddings are in fact a class of techniques where individual words are represented as real-valued vectors in a predefined vector space. Each word is mapped to one vector and the vector values are learned in a way that resembles a neural network, and hence the technique is often lumped into the field of deep learning.

Key benefits of word embeddings include:

- **Dimensionality Reduction:** Word embeddings reduce the dimensionality of text data, making it easier to process.
- **Semantic Similarity:** Words that are semantically similar are close to each other in the vector space, making it possible to capture semantic relationships between words.
- **Contextual Information:** Word embeddings capture the context in which words are used, allowing the model to learn not just the words but also their meaning in the text.

![](https://aiml.com/wp-content/uploads/2023/02/disadvantage-bow-1024x650.png)

__Image Credit to https://aiml.com/what-are-the-advantages-and-disadvantages-of-bag-of-words-model/__

## Bag of Words (BoW): A Simple Start

The Bag of Words (BoW) model was one of the earliest attempts to represent text in a way that computers could understand. This method converts text into a ‘bag’ of words, focusing only on the frequency of words while ignoring their order. While BoW is remarkably simple, its limitations are clear. It strips away context and meaning, treating words like "cat" and "dog" as no more related than "airplane" and "bicycle." Despite these drawbacks, BoW became a staple in early text classification tasks like spam filtering and sentiment analysis.

The limitations of BoW became more obvious as language models became essential for applications like chatbots and machine translation, where nuance and word relationships are crucial. Nevertheless, it laid the groundwork for more sophisticated models to come.

### 1. Bag of Words (BoW)

BoW simply represents a document as a frequency count of words. Here's an example using `CountVectorizer` from `sklearn`.

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = ["I love deep learning", "Deep learning is fun", "I love NLP"]

# Convert to BoW representation
vectorizer = CountVectorizer()
bow_rep = vectorizer.fit_transform(documents)

# Output BoW matrix
print(bow_rep.toarray())
print(vectorizer.get_feature_names_out())
```

```
[[1 0 0 1 1 0]
 [1 1 1 1 0 0]
 [0 0 0 0 1 1]]
['deep' 'fun' 'is' 'learning' 'love' 'nlp']
```

**Explanation:**  
- `CountVectorizer` converts the text documents into a sparse matrix where each row represents a document and each column represents a word in the vocabulary.
- This basic BoW model doesn't consider word order or semantics.

**Key points to remember about CountVectorizer:**

- Vocabulary creation: It automatically creates a vocabulary of unique words from the input text.
- Sparse matrix representation: It represents the BoW representation as a sparse matrix, which is efficient for handling large datasets with many unique words.
- Customization: You can customize the CountVectorizer using various parameters, such as stop_words, ngram_range, and max_features, to tailor the BoW representation to your specific needs.

A sparse matrix is a matrix where most of the elements are zero. This is often the case in datasets with high dimensionality, such as those encountered in natural language processing or machine learning.

In the context of the Bag-of-Words (BoW) representation, a sparse matrix is used to represent the occurrence of words in a document. Each row of the matrix corresponds to a document, and each column corresponds to a unique word in the vocabulary. The value at the intersection of a row and column indicates the number of times that word appears in the corresponding document.

Because most documents will only contain a small subset of the total vocabulary, the resulting matrix will be very sparse, with many zero values. This makes it inefficient to store the entire matrix in a dense format. Instead, sparse matrix representations are used to store only the non-zero elements of the matrix, along with their corresponding row and column indices. This can significantly reduce the memory footprint of the BoW representation, making it feasible to work with large datasets.

There are several different sparse matrix representations, including:

* **Compressed Sparse Row (CSR) format:** This format stores the matrix as three arrays: one for the non-zero values, one for the column indices of the non-zero values, and one for the row offsets of the non-zero values.
* **Compressed Sparse Column (CSC) format:** This format is similar to CSR, but it stores the matrix by column instead of by row.
* **Coordinate List (COO) format:** This format stores the matrix as three arrays: one for the row indices of the non-zero values, one for the column indices of the non-zero values, and one for the non-zero values themselves.

The choice of which sparse matrix representation to use depends on the specific application and the type of operations that will be performed on the matrix.

### Limitations of Bag of Words:

While the BoW model is simple and easy to implement, it does have its limitations:

- It doesn't capture the order of words or the context in which they're used. This can be problematic for tasks where the order of words is important, such as sentiment analysis or machine translation.
- It treats each word as an independent entity and doesn't capture the semantic relationships between words.
- It can result in a high-dimensional feature vector due to the large size of the vocabulary, which can lead to computational challenges.

## Continuous Bag of Words (CBOW): The Quest for Context

To address the lack of context in BoW, researchers introduced the Continuous Bag of Words (CBOW) model. Part of the Word2Vec family, CBOW aims to produce word embeddings, vectorized representations that encode semantic meaning. Unlike BoW, CBOW takes surrounding words (context) into account to predict a target word. This is a significant leap forward, allowing machines to better understand the relationships between words, although CBOW still struggles with subtle differences like polysemy (where a word has multiple meanings).

With models like CBOW, AI began to move closer to understanding text in a way more akin to how humans do—grasping word meanings based on their surroundings. This breakthrough became a springboard for further advances in text representation.

```python
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
```

```
Embedding for 'winter': [-0.41915172 -0.623203   -0.82449454  2.5393567   0.4014106  -1.2217898
  0.8545555   1.1546286  -1.6388881   1.0636564 ]
Embedding for 'summer': [ 0.27508304  0.14038266 -0.60545456  2.6588614  -0.67323846 -0.41758934
 -1.9973164  -0.25613934  0.6041437   0.02313808]
Embedding for 'hot': [ 1.5088211   0.11288307 -2.1109881   1.1869487  -0.72843784  0.39130706
 -1.167067    0.02461074  1.6819512   0.893444  ]
Embedding for 'cold': [ 0.46896353  0.09806713  0.886707   -2.3495743  -1.8937181  -1.7938056
  0.5059122   0.73013675  0.63746464  0.454989  ]
Embedding for 'snow': [-0.98443097  0.36648035 -1.244046    0.8789777  -0.73171115 -0.8969204
 -0.6760932   1.4665215  -1.4990703  -1.3799495 ]
Embedding for 'sun': [-0.7282811  -0.45672363 -1.2333208   1.1166155  -0.8474363   1.9662368
 -1.5384043   0.01905673  0.08274112 -0.20752785]
['winter', 'snow']
Predicted next word: cold
```

The Continuous Bag of Words (CBOW) model is a popular method for training word embeddings. Word embeddings are a type of word representation that allows words with similar meaning to have a similar representation, which is crucial for many natural language processing (NLP) tasks. Word embeddings generated by CBOW can be used to measure semantic similarity between words or documents. For example, you can calculate the cosine similarity between two word vectors to determine how similar their meanings are.


## Word2Vec: The Revolution Before the Revolution

The Word2Vec models, including CBOW, marked a seismic shift in NLP. Developed by Google in 2013, these shallow neural networks could produce high-quality word embeddings efficiently. For the first time, NLP models could understand relationships between words based on their usage patterns across vast datasets. Word2Vec helped NLP models grasp concepts like the famous "king - man + woman = queen" analogy, encapsulating the semantic relationship between words in vector space. 

Yet, despite their power, Word2Vec models had limitations. They were still relatively shallow, meaning they struggled with tasks that required a deep understanding of word order, syntax, or polysemy. This set the stage for more complex models that could tackle those challenges head-on.

## Conclusion: The Future Built on the Past

In 2024, as AI begins to reshape industries, politics, and daily life, it’s easy to forget the early days of NLP when methods like BoW dominated the field. Understanding the evolution of text representation from these simpler models to GPT and beyond — helps us appreciate the complex AI systems we rely on today. As we look to the future, where NLP models could enable even more personalized and nuanced interactions between humans and machines, one thing is clear: the past is prologue.

The next wave of AI innovation will build upon these foundations, and it’s crucial to understand where we’ve come from to anticipate where we’re headed.
