# Offensive Text Classification Model

A machine learning solution for automated content moderation that classifies social media posts into three categories: hate speech, offensive language, or neither. This system provides real-time content filtering capabilities to help maintain a safe and welcoming online environment.

## 🎯 Project Overview

This project addresses the critical challenge of moderating large volumes of user-generated content on social media platforms. With millions of daily posts, manual review becomes impractical and ineffective. Our AI-powered solution automatically analyzes text content and categorizes it before it becomes visible to other users.

### Problem Statement
- Automated bots posting hate speech and inflammatory content
- Overwhelming volume making manual moderation impossible
- Need for real-time content filtering to maintain platform safety
- Requirement to distinguish between different levels of offensive content

### Solution
An intelligent text classification system that:
- Processes thousands of posts per second
- Learns language patterns to identify harmful content
- Provides probability scores for classification confidence
- Includes interactive testing capabilities

## 🏗️ Architecture

### Classification Categories
- **Class 0**: Hate Speech - Content containing explicit hate targeting individuals or groups
- **Class 1**: Offensive Language - General profanity or inappropriate content
- **Class 2**: Neither - Neutral or positive content

### Machine Learning Pipeline
1. **Text Preprocessing** - Clean and prepare raw text data
2. **Feature Extraction** - Convert text to numerical vectors using TF-IDF
3. **Model Training** - Train logistic regression classifier
4. **Prediction** - Generate classifications with confidence scores
5. **Evaluation** - Assess performance using multiple metrics

## 🚀 Features

- **Real-time Classification** - Instant analysis of new text inputs
- **Probability Scoring** - Confidence levels for each prediction
- **Interactive GUI** - User-friendly interface for testing
- **Comprehensive Visualization** - Multiple charts for data analysis
- **Model Interpretability** - Clear insights into decision-making process

## 📋 Requirements

```python
pandas
matplotlib
seaborn
scikit-learn
numpy
ipywidgets
```

## 🔧 Installation

1. Clone the repository or download `model.py`
2. Install required dependencies:
```bash
pip install pandas matplotlib seaborn scikit-learn numpy ipywidgets
```
3. Ensure your dataset is available at `./data/train.csv`

## 📊 Dataset Format

The model expects a CSV file with the following structure:
```
tweet,class
"Example text content",0
"Another example",1
```

**Columns:**
- `tweet`: Text content to classify
- `class`: Integer label (0=Hate Speech, 1=Offensive, 2=Neither)

## 🎮 Usage

### Basic Classification
```python
# Load and prepare the model (run the training sections first)
test_text = "Your text here"
text_tfidf = tfidf.transform([test_text])
prediction = model.predict(text_tfidf)[0]
probability = model.predict_proba(text_tfidf)[0]
```

### Interactive Testing
The model includes an interactive widget interface for real-time testing:
- Enter text in the provided textarea
- Click "Analyze Text" to get instant classification
- View detailed probability breakdown for all categories

### Batch Processing
```python
# For multiple texts
texts = ["Text 1", "Text 2", "Text 3"]
texts_tfidf = tfidf.transform(texts)
predictions = model.predict(texts_tfidf)
probabilities = model.predict_proba(texts_tfidf)
```

## 📊 Model Performance

- **Algorithm**: Logistic Regression with TF-IDF vectorization
- **Accuracy**: ~89% on test dataset
- **Features**: 5,000 TF-IDF features with English stop words removal
- **Training**: 80/20 train-test split with stratified sampling

### Performance Metrics
- Confusion matrix analysis for detailed performance breakdown
- Individual class accuracy assessment
- Visual analysis through t-SNE dimensionality reduction

## 📈 Visualizations

The model generates three key visualizations:

1. **Label Distribution Chart** - Shows the balance of categories in the dataset
2. **t-SNE Visualization** - 2D representation of high-dimensional text embeddings
3. **Confusion Matrix** - Detailed performance analysis across all categories

## 🔍 Model Details

### TF-IDF Vectorization
- **Max Features**: 5,000 most important words
- **Stop Words**: English stop words removed
- **Approach**: Term Frequency-Inverse Document Frequency weighting

### Logistic Regression
- **Max Iterations**: 1,000 for convergence
- **Multi-class Strategy**: One-vs-rest approach
- **Output**: Both class predictions and probability distributions

### Why This Approach?
- **Linear models** excel with high-dimensional sparse text data
- **Fast training and prediction** suitable for real-time applications
- **Interpretable results** allow examination of feature importance
- **Probabilistic output** provides confidence scoring

## ⚡ Performance Characteristics

- **Training Speed**: Fast convergence with 1,000 max iterations
- **Prediction Speed**: Real-time classification capability
- **Memory Usage**: Efficient with sparse matrix representation
- **Scalability**: Handles large volumes of text data

## 🎯 Use Cases

- **Social Media Platforms** - Automated content moderation
- **Comment Systems** - Filter inappropriate comments
- **Forum Moderation** - Pre-screen user posts
- **Chat Applications** - Real-time message filtering
- **Content Review** - Assist human moderators with initial screening

## ⚠️ Limitations

- **Context Understanding** - May struggle with sarcasm or nuanced context
- **Evolving Language** - Requires retraining for new slang or terminology
- **Cultural Bias** - Performance dependent on training data composition
- **Edge Cases** - Ambiguous content may require human review
