# Fraud Detection - Ensemble Learning

A Machine Learning project for detecting fraudulent financial transactions using multiple classification algorithms and ensemble learning techniques.

The project builds and compares different Machine Learning models on a fraudulent transaction dataset and evaluates their performance using classification accuracy.

## Problem Statement

A financial institution wants to identify potentially fraudulent transactions.

The dataset contains transaction-related information such as:

- Transaction Amount
- Transaction Time
- Account Age
- Number of Previous Transactions
- Location Difference
- Device Type
- Failed Login Attempts

The target variable is:

- `0` → Normal Transaction
- `1` → Fraudulent Transaction

## Objective

The main objective of this project is to investigate and compare multiple Machine Learning and ensemble learning techniques for fraud detection.

The following models are implemented:

1. Decision Tree Classifier
2. Random Forest Classifier
3. Bagging Classifier
4. AdaBoost Classifier
5. Hard Voting Classifier
6. Soft Voting Classifier

The performance of the models is compared to understand which approach provides better classification results.

## Machine Learning Models

### 1. Decision Tree Classifier

A Decision Tree is used as an individual classification model to establish a baseline for comparison.

### 2. Random Forest Classifier

Random Forest combines multiple decision trees and uses their collective predictions to improve the robustness of the model.

### 3. Bagging Classifier

Bagging creates multiple models using different samples of the training data and combines their predictions.

In this implementation, Logistic Regression is used as the base estimator.

### 4. AdaBoost Classifier

AdaBoost combines multiple weak learners sequentially, giving more importance to incorrectly classified samples during training.

Logistic Regression is used as the base estimator in this implementation.

### 5. Hard Voting Classifier

Hard Voting combines:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors

The final prediction is based on the majority vote of the individual classifiers.

### 6. Soft Voting Classifier

Soft Voting also combines:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors

Instead of using only predicted classes, it uses the predicted probabilities of the classifiers to make the final decision.

## Project Workflow

The project follows an end-to-end Machine Learning workflow:

```text
Load Dataset
      ↓
Explore Dataset
      ↓
Check Missing Values
      ↓
Separate Features and Target
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Train Multiple Models
      ↓
Generate Predictions
      ↓
Evaluate Model Accuracy
      ↓
Compare Model Performance



Project Structure

Fraud-Detection-Ensemble-Learning-Repo/
│
├── AdaBoost-Classifier/
│   ├── Fraud-Detection-AdaBoost.py
│   └── Fraudulent_Transaction_Detection.csv
│
├── All-Ensemble-Models/
│   ├── Fraud-All-Models.py
│   └── Fraudulent_Transaction_Detection.csv
│
├── Bagging-Classifier/
│   ├── Fraud-Detection-Bagging.py
│   └── Fraudulent_Transaction_Detection.csv
│
├── Decision-Tree-Classifier/
│   ├── Fraud-Detection-DT.py
│   └── Fraudulent_Transaction_Detection.csv
│
├── Random-Forest-Classifier/
│   ├── Fraud-Detection-RF.py
│   └── Fraudulent_Transaction_Detection.csv
│
├── Voting-Classifier/
│   ├── Fraud-Detection-Hard-Voting.py
│   ├── Fraud-Detection-Soft-Voting.py
│   └── Fraudulent_Transaction_Detection.csv
│
├── Fraudulent_Transaction_Detection.csv
├── README.md
├── requirements.txt
└── .gitignore
