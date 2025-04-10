# 🚀 UPI Fraud Detection

This project aims to detect fraudulent transactions in Unified Payments Interface (UPI) systems using machine learning techniques. It leverages advanced models, handles class imbalance, and evaluates performance using multiple metrics.

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Dataset](#-dataset)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Models and Evaluation](#-models-and-evaluation)
- [Results](#-results)

---

## 🌟 About the Project

Fraudulent transactions in UPI systems pose a significant threat to financial security. This project builds a machine learning pipeline to classify fraudulent transactions effectively, addressing class imbalance and providing robust evaluation.

### 🔑 Key Highlights:

- **Data Preprocessing**: Handles missing values, one-hot encoding, and feature scaling.
- **Class Imbalance Handling**: Uses SMOTE to oversample the minority class.
- **Model Training**: Implements Random Forest, XGBoost, Logistic Regression, Decision Tree, and Gaussian Naive Bayes.
- **Evaluation**: Uses confusion matrices, classification reports, and accuracy scores.

---

## 📊 Dataset

The dataset is sourced from Kaggle: **Online Payments Fraud Detection Dataset**

### Dataset Overview:
- **step**: Time step of the transaction.
- **type**: Transaction type (e.g., CASH-IN, CASH-OUT).
- **amount**: Transaction amount.
- **oldbalanceOrg / newbalanceOrig**: Origin account balances before and after the transaction.
- **oldbalanceDest / newbalanceDest**: Destination account balances before and after the transaction.
- **isFraud**: Target label (1 = Fraudulent, 0 = Genuine).

---

## 🎯 Features

### Data Preprocessing:
- Dropped irrelevant columns: `nameOrig`, `nameDest`
- Applied one-hot encoding to categorical feature: `type`
- Scaled numerical features using `StandardScaler`

### Class Imbalance Handling:
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) to balance classes

### Model Training:
- Trained the following classifiers:
  - Random Forest
  - XGBoost
  - Logistic Regression
  - Decision Tree
  - Gaussian Naive Bayes

### Evaluation Metrics:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Accuracy Score

---

## 🛠️ Tech Stack

| Component       | Technology                             |
|----------------|-----------------------------------------|
| Language        | Python                                  |
| Libraries       | Scikit-learn, XGBoost, Imbalanced-learn |
| Visualization   | Matplotlib, Seaborn                     |
| Data Processing | Pandas, NumPy                           |

---

## ⚙️ Installation

### Prerequisites:
- Python 3.8+
- Jupyter Notebook or Google Colab

### Step 1: Install Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
Step 2: Download the Dataset from Kaggle
bash
Copy
Edit
pip install kaggle
kaggle datasets download -d rupakroy/online-payments-fraud-detection-dataset
unzip online-payments-fraud-detection-dataset.zip
🚀 Usage
Open UPI-Fraud-Detection.ipynb in Jupyter Notebook or Google Colab.

Execute the following steps:
Load and preprocess data

Apply SMOTE for class balancing

Train and evaluate machine learning models

📈 Models and Evaluation
✅ Models Used:
Random Forest

XGBoost

Logistic Regression

Decision Tree

Gaussian Naive Bayes

📌 Evaluation Criteria:
Confusion Matrix

Classification Report (Precision, Recall, F1-score)

Accuracy Score

📊 Results
🏆 Decision Tree achieved the highest accuracy: 99.97% with balanced precision and recall.

✅ Random Forest showed good performance, though with slightly lower precision.

⚠️ Gaussian Naive Bayes underperformed due to its assumptions about data distribution.

