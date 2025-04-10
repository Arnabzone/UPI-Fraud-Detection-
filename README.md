🚀 UPI Fraud Detection
This project aims to detect fraudulent transactions in Unified Payments Interface (UPI) systems using machine learning techniques. It leverages advanced models, handles class imbalance, and evaluates performance using multiple metrics.

📋 Table of Contents
About the Project

Dataset

Features

Tech Stack

Installation

Usage

Models and Evaluation

Results

🌟 About the Project
Fraudulent transactions in UPI systems are a major challenge in financial security. This project builds a machine learning pipeline to classify fraudulent transactions effectively, addressing class imbalance and providing robust evaluation.

Key Highlights:
Data Preprocessing: Handles missing values, applies one-hot encoding, and scales features.

Class Imbalance Handling: Uses SMOTE to oversample the minority class.

Model Training: Implements Random Forest, XGBoost, Logistic Regression, Decision Tree, and Gaussian Naive Bayes.

Evaluation: Includes confusion matrices, classification reports, and accuracy scores.

📊 Dataset
The dataset used is from Kaggle:
Online Payments Fraud Detection Dataset.

Dataset Overview:
The dataset contains information about online transactions with features such as:

step: Time step of the transaction.

type: Transaction type (e.g., CASH-IN, CASH-OUT).

amount: Transaction amount.

oldbalanceOrg/newbalanceOrig: Balances before and after the transaction for the origin account.

oldbalanceDest/newbalanceDest: Balances before and after the transaction for the destination account.

isFraud: Target variable indicating if the transaction is fraudulent.

🎯 Features
Data Preprocessing:

Dropped irrelevant columns (nameOrig, nameDest).

Applied one-hot encoding for categorical variables like type.

Scaled numerical features using StandardScaler.

Class Imbalance Handling:

Used SMOTE to oversample the minority class (isFraud).

Model Training:

Trained multiple classifiers including Random Forest, XGBoost, Logistic Regression, Decision Tree, and Gaussian Naive Bayes.

Evaluation Metrics:

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

Accuracy Score

🛠️ Tech Stack
Component	Technology
Language	Python
Libraries/Frameworks	Scikit-learn, XGBoost, Imbalanced-learn
Visualization	Matplotlib, Seaborn
Data Processing	Pandas, NumPy
⚙️ Installation
Prerequisites
Ensure you have Python 3.8+ installed along with Jupyter Notebook or Google Colab access.

Steps

Install required libraries:

bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
Download the dataset from Kaggle:

bash
!pip install kaggle
!kaggle datasets download -d rupakroy/online-payments-fraud-detection-dataset
!unzip online-payments-fraud-detection-dataset.zip
🚀 Usage
Open the notebook UPI-Fraud-Detection.ipynb in Jupyter Notebook.

Run each cell sequentially to:

Load and preprocess data.

Apply SMOTE for class imbalance.

Train and evaluate models.

📈 Models 
The following models were trained and evaluated:

Random Forest

XGBoost

Logistic Regression

Decision Tree

Gaussian Naive Bayes

Evaluation Metrics:
Each model was evaluated using:

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

Accuracy Score

📊 Results

->Decision Tree achieved the highest accuracy (99.97%) with balanced precision and recall.

->Random Forest performed well but had lower precision compared to recall.

->Gaussian Naive Bayes struggled due to its assumptions about data distribution.
