# 📄 Email Spam Detection using Logistic Regression

A machine learning-based spam email classifier built using Logistic Regression. This project processes multiple datasets, performs preprocessing and vectorization with TF-IDF, trains a logistic regression model, and allows custom input testing. Achieves over 96% accuracy.

---

## 📊 Features

* Combines multiple email datasets for diverse spam/ham examples
* Preprocessing includes:

  * HTML tag removal
  * Lowercasing
  * Punctuation and digit cleaning
  * URL/email stripping
* TF-IDF vectorization (3000 top features)
* Logistic Regression model with high accuracy
* Real-time CLI prediction from user input
* Model and vectorizer persistence using `joblib`

---

## 📁 Project Structure

```
.
├── datasets/
│   ├── combined_data.csv
│   ├── emails.csv
│   ├── spam.csv
│   └── spam_assassin.csv
├── merge.py              # Merges and formats datasets
├── preprocess.py         # Handles all text preprocessing
├── train.py              # Trains and saves the model
├── predict_custom.py     # Takes user input and predicts spam/ham
└── README.md
```

---

## ⚙️ Requirements

```bash
pip install pandas scikit-learn beautifulsoup4 joblib
```

---

## 🚀 Getting Started

### 1. Train the model

```bash
python train.py
```

### 2. Predict on custom input

```bash
python predict_custom.py
```

---

## 🌐 Dataset Sources

* `emails.csv` from Kaggle
* `spam.csv` from UCI ML Repository
* `spam_assassin.csv` public corpora
* Combined for improved generalization

---

## 🌟 Accuracy

* Achieved **96%+** accuracy on the merged dataset
* Evaluated using precision, recall, F1-score

---


## ✊ Built With

* Python 3.10+
* scikit-learn
* pandas
* BeautifulSoup
* joblib
