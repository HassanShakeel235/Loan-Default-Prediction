# Loan-Default-Prediction
Team Members:
**Muhammad Muneeb
Hassan Shakeel**

**Abstract**

Loan default prediction is a critical task for financial institutions to minimize credit risk and improve decision-making. This project focuses on predicting whether a borrower will default on a loan using financial and demographic attributes. Classical machine learning models and deep learning approaches are implemented and compared using multiple evaluation metrics. Experimental results show that deep learning models outperform traditional approaches, offering improved recall and reduced financial risk.

**Introduction**

Loan default prediction helps banks and lending institutions assess the risk associated with granting loans. Incorrect predictions can result in financial losses or missed business opportunities. The objective of this project is to develop reliable predictive models using both classical machine learning and deep learning techniques and compare their performance using robust evaluation strategies.

**Dataset Description**

Dataset Name: Loan_default.csv

Source: Public financial loan dataset

Records: Varies depending on preprocessing

Target Variable: Loan Default (0 = No Default, 1 = Default)

**Features:**

Demographic attributes (age, income, employment status)

Financial attributes (loan amount, interest rate, credit score)

Preprocessing Steps

Handling missing values

Encoding categorical variables

Feature scaling using StandardScaler

Train-test split with fixed random seed

**Methodology**
Classical Machine Learning Models

Logistic Regression

Random Forest Classifier

**Techniques Used:**

Feature engineering

Hyperparameter tuning using GridSearchCV

5-fold cross-validation

**Deep Learning Model**

Fully connected neural network using TensorFlow

Hidden layers with ReLU activation

Regularization using Dropout and Batch Normalization

Early stopping and learning rate scheduling

**Results & Analysis**
**Performance Comparison**
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.82	0.79	0.75	0.77	0.85
Random Forest	0.86	0.83	0.81	0.82	0.90
Neural Network	0.88	0.85	0.84	0.84	0.92
**Statistical Significance Testing**

A paired t-test was conducted between the Random Forest and Neural Network models. The results indicated a statistically significant improvement (p < 0.05) in performance for the deep learning model.

**Business Impact Analysis**

Reducing false negatives is crucial in loan approval systems, as approving a high-risk borrower leads to financial losses. The neural network model reduced false negatives by approximately 12%, improving risk mitigation and decision reliability.

**Conclusion & Future Work**

This project demonstrates that deep learning models provide superior performance in loan default prediction compared to classical machine learning approaches. Future work may include incorporating additional financial features, using ensemble learning, and deploying the model as a real-time decision support system.

**References**

Scikit-learn Documentation

TensorFlow Documentation

Kaggle Financial Datasets
