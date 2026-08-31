# Customer Loan Approval Prediction

A Machine Learning classification project that predicts whether a customer's loan will be approved based on customer financial and personal information.

The project compares multiple classification algorithms and uses Hard Voting and Soft Voting Ensemble techniques to improve and compare prediction performance.

## Dataset

The dataset contains the following columns:

- Age
- Income
- CreditScore
- ExistingLoan
- EmploymentExperience
- LoanAmount
- LoanApproved

### Target Variable

`LoanApproved`

The target represents whether the customer's loan is approved.

## Machine Learning Models

The following classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN) Classifier

The project also implements:

4. Hard Voting Classifier
5. Soft Voting Classifier

## Project Workflow

```text
Load Dataset
      ↓
Exploratory Data Analysis
      ↓
Check Missing Values
      ↓
Separate Features and Target
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Train Individual Models
      ↓
Calculate Accuracy
      ↓
Hard Voting
      ↓
Soft Voting
      ↓
Compare Model Performance