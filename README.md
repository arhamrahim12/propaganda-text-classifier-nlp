# Propaganda Detection and Technique Classification in Text

This project explores binary and multi-class classification of propaganda in text using machine learning and deep learning. Models were tested on annotated datasets from the Propaganda Techniques Corpus.

## 🎯 Objectives

- Detect whether a sentence contains propaganda (binary)
- Classify the technique used (multi-class: 8 propaganda types)

## 📦 Datasets

- Source: [Propaganda Techniques Corpus](https://propaganda.qcri.org/semeval2020-task11/)
- Format: TSV files with sentence and label pairs
- Classes include: loaded language, flag waving, repetition, doubt, causal simplification, and more

## 🧠 Methods

### Task 1: Binary Classification (Propaganda vs Not)
- **Naive Bayes with TF-IDF**
- **Logistic Regression with TF-IDF**
- Feature extraction: TF-IDF (unigram–trigram)
- Preprocessing: custom spaCy pipeline + SMOTE for imbalance

### Task 2: Multi-Class Technique Classification
- **Multinomial Logistic Regression** with:
  - Word2Vec
  - TF-IDF
  - Combined Word2Vec + TF-IDF
- **LSTM** with Word2Vec embeddings
- Preprocessing: spaCy-based POS + NER tagging

## ⚙️ Preprocessing Pipeline

- Lowercasing, stopword removal
- POS tagging and entity recognition
- Custom transformer pipelines (Scikit-learn compatible)
- Word embeddings (Word2Vec trained from scratch)

## 📊 Model Performance

| Task | Model | Test Accuracy |
|------|-------|----------------|
| Task 1 | Logistic Regression | **69%** |
| Task 1 | Naive Bayes | 65% |
| Task 2 | Word2Vec + TF-IDF | **47%** |
| Task 2 | LSTM + Word2Vec | 40% |

## 📁 Project Structure

├── report.pdf # Full report with results ├── code.pdf # All code used ├── predictions.csv # Model outputs ├── README.md

markdown
Copy code

## 📚 Libraries Used

- scikit-learn
- imbalanced-learn (SMOTE)
- spaCy, NLTK
- Gensim (Word2Vec)
- TensorFlow/Keras (LSTM)

## 🚀 Future Work

- BERT-based classification
- Better class balancing for rare techniques
- Ensemble model experiments
