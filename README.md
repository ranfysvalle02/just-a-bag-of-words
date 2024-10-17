# preBERT: The Foundations of Text Representation

**The World Before BERT: How Early Text Representation Models Shaped Today’s AI**

**Introduction:**

In the last few years, models like BERT (Bidirectional Encoder Representations from Transformers) have radically transformed how we interact with technology, from search engines to virtual assistants. Yet, behind BERT's incredible understanding of language lies a long history of evolving text representation techniques that have shaped the foundation of modern Natural Language Processing (NLP). This blog post takes a journey through the pioneering approaches that dominated before BERT’s emergence, exploring the humble beginnings of Bag of Words (BoW), the innovations of Word2Vec, and how these earlier methods helped us build the AI systems that today power some of the world's largest companies.

As artificial intelligence (AI) continues to integrate deeper into industries from healthcare to finance, understanding these historical methods isn't just academic—it's essential for grasping where NLP is heading in the future. 

**Bag of Words (BoW): A Simple Start**

The Bag of Words (BoW) model was one of the earliest attempts to represent text in a way that computers could understand. This method converts text into a ‘bag’ of words, focusing only on the frequency of words while ignoring their order. While BoW is remarkably simple, its limitations are clear. It strips away context and meaning, treating words like "cat" and "dog" as no more related than "airplane" and "bicycle." Despite these drawbacks, BoW became a staple in early text classification tasks like spam filtering and sentiment analysis.

The limitations of BoW became more obvious as language models became essential for applications like chatbots and machine translation, where nuance and word relationships are crucial. Nevertheless, it laid the groundwork for more sophisticated models to come.

### 1. **Bag of Words (BoW)**

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

**Explanation:**  
- `CountVectorizer` converts the text documents into a sparse matrix where each row represents a document and each column represents a word in the vocabulary.
- This basic BoW model doesn't consider word order or semantics.


**Continuous Bag of Words (CBOW): The Quest for Context**

To address the lack of context in BoW, researchers introduced the Continuous Bag of Words (CBOW) model. Part of the Word2Vec family, CBOW aims to produce word embeddings, vectorized representations that encode semantic meaning. Unlike BoW, CBOW takes surrounding words (context) into account to predict a target word. This is a significant leap forward, allowing machines to better understand the relationships between words, although CBOW still struggles with subtle differences like polysemy (where a word has multiple meanings).

With models like CBOW, AI began to move closer to understanding text in a way more akin to how humans do—grasping word meanings based on their surroundings. This breakthrough became a springboard for further advances in text representation.

### 2. **Continuous Bag of Words (CBOW) - PyTorch Lightning**

For CBOW, the context (surrounding words) predicts the target word. Here's a simple PyTorch Lightning implementation.

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

# Example dataset class
class CBOWDataset(Dataset):
    def __init__(self, context_data, target_data):
        self.context_data = context_data
        self.target_data = target_data

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, idx):
        return self.context_data[idx], self.target_data[idx]

# Example CBOW model
class CBOWModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        embedded = self.embeddings(context).mean(dim=1)  # Average context embeddings
        output = self.linear(embedded)
        return output

    def training_step(self, batch, batch_idx):
        context, target = batch
        output = self.forward(context)
        loss = nn.CrossEntropyLoss()(output, target)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# Sample training loop (context_data and target_data should be prepared beforehand)
context_data = torch.tensor([[1, 2], [3, 4], [2, 3]])
target_data = torch.tensor([0, 3, 2])

train_dataset = CBOWDataset(context_data, target_data)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

model = CBOWModel(vocab_size=5, embedding_dim=10)
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model, train_loader)
```

**Explanation:**  
- CBOW predicts a word based on its surrounding words (context). The model averages the embeddings of the context words to predict the target word.

**Skip-Gram: Predicting Context for Precision**

If CBOW was a step forward, the Skip-Gram model took NLP even further. Rather than predicting a target word from its context, Skip-Gram reverses the process, using a given word to predict its surrounding words. This model is particularly useful for understanding rare words and phrases, outperforming CBOW when data is limited.

Skip-Gram’s success in capturing intricate relationships between words made it a favorite in building large-scale models at organizations like Google and Facebook, allowing them to create better recommendation systems, improve search engine results, and enhance voice recognition technology. 

### 3. **Skip-Gram - PyTorch Lightning**

Skip-Gram does the reverse of CBOW, predicting the surrounding context based on a target word.

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# Skip-Gram Dataset
class SkipGramDataset(Dataset):
    def __init__(self, target_data, context_data):
        self.target_data = target_data
        self.context_data = context_data

    def __len__(self):
        return len(self.context_data)

    def __getitem__(self, idx):
        return self.target_data[idx], self.context_data[idx]

# Skip-Gram Model
class SkipGramModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target):
        embedded = self.embeddings(target)
        output = self.linear(embedded)
        return output

    def training_step(self, batch, batch_idx):
        target, context = batch
        output = self.forward(target)
        loss = nn.CrossEntropyLoss()(output, context)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# Example data for Skip-Gram (target and context should be prepared beforehand)
target_data = torch.tensor([1, 2, 3])
context_data = torch.tensor([0, 2, 4])

train_dataset = SkipGramDataset(target_data, context_data)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

model = SkipGramModel(vocab_size=5, embedding_dim=10)
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model, train_loader)
```

**Explanation:**  
- The Skip-Gram model predicts the surrounding words (context) given a target word.
- Like CBOW, the training uses a neural network to generate word embeddings, but the task is reversed.

**Word2Vec: The Revolution Before the Revolution**

The Word2Vec models, including CBOW and Skip-Gram, marked a seismic shift in NLP. Developed by Google in 2013, these shallow neural networks could produce high-quality word embeddings efficiently. For the first time, NLP models could understand relationships between words based on their usage patterns across vast datasets. Word2Vec helped NLP models grasp concepts like the famous "king - man + woman = queen" analogy, encapsulating the semantic relationship between words in vector space. 

Yet, despite their power, Word2Vec models had limitations. They were still relatively shallow, meaning they struggled with tasks that required a deep understanding of word order, syntax, or polysemy. This set the stage for more complex models that could tackle those challenges head-on.

### 4. **Word2Vec (Skip-Gram variant) - PyTorch Lightning**

Word2Vec is a family of models that includes CBOW and Skip-Gram. Here’s a simplified Word2Vec (Skip-Gram variant) using PyTorch Lightning.

```python
class Word2Vec(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, words):
        return self.embeddings(words)

    def training_step(self, batch, batch_idx):
        target, context = batch
        target_embed = self.forward(target)
        context_embed = self.forward(context)

        # Maximize the dot product between target and context embeddings
        dot_product = torch.bmm(target_embed.unsqueeze(1), context_embed.unsqueeze(2)).squeeze()
        loss = -torch.log(torch.sigmoid(dot_product)).mean()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# Sample Word2Vec Skip-Gram style training data
target_data = torch.tensor([1, 2, 3])
context_data = torch.tensor([0, 2, 4])

train_dataset = SkipGramDataset(target_data, context_data)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

model = Word2Vec(vocab_size=5, embedding_dim=10)
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model, train_loader)
```

**Explanation:**  
- Word2Vec (Skip-Gram) uses word embeddings to capture semantic and syntactic similarities between words.
- The model maximizes the dot product between target and context embeddings, pushing similar words closer together in vector space.

**Conclusion: The Future Built on the Past**

In 2024, as AI begins to reshape industries, politics, and daily life, it’s easy to forget the early days of NLP when methods like BoW or Skip-Gram dominated the field. Understanding the evolution of text representation from these simpler models to BERT—and now, beyond—helps us appreciate the complex AI systems we rely on today. As we look to the future, where NLP models could enable even more personalized and nuanced interactions between humans and machines, one thing is clear: the past is prologue.

The next wave of AI innovation will build upon these foundations, and it’s crucial to understand where we’ve come from to anticipate where we’re headed. With new breakthroughs in large language models, multimodal learning, and AI ethics, the journey that began with simple bags of words is far from over.


