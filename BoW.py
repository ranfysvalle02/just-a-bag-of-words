from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = ["I love deep learning", "Deep learning is fun", "I love NLP"]

# Convert to BoW representation
vectorizer = CountVectorizer()
bow_rep = vectorizer.fit_transform(documents)

# Output BoW matrix
print(bow_rep.toarray())
print(vectorizer.get_feature_names_out())
